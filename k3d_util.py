import numpy as np
from PIL import Image
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from scipy import interpolate
import base64, io, time, datetime, sys
import threading
from frame_writer2 import FFmpegFrameWriter
import json

# 定数の定義 ##########

# 対物レンズ倍率
MAGNIFICATION = 40

# 撮影時のビニング
BINNING = 1

# Z間隔
Z_SLICE_LENGTH = 2.0

# 画像の画素数を減らすための画像縮小率
REDUCE_RATIO = 4

# Zスタックの枚数を減らすためのZステップ間隔
Z_STEP = 1

# カメラの画素サイズ(通常は正方形)
CAMERA_PIXEL_SIZE = 6.5

# color_map: A list of float quadruplets (attribute value, R, G, B), sorted by attribute value. The first
#            quadruplet should have value 0.0, the last 1.0; R, G, B are RGB color components in the range 0.0 to 1.0.
# 最大4ボリューム分まで定義可能
COLOR_MAP_LIST = [
    [0.0, 0.0, 0.0, 1.0,  1.0, 0.0, 0.0, 1.0],  # 青
    [0.0, 0.0, 1.0, 0.0,  1.0, 0.0, 1.0, 0.0],  # 緑
    [0.0, 0.8, 0.6, 0.0,  1.0, 0.8, 0.6, 0.0],  # 橙
    [0.0, 1.0, 0.0, 0.0,  1.0, 1.0, 0.0, 0.0],  # 赤
]

# 動画作成処理の負荷低減用スリープ時間(単位: 秒)
MOVIE_WAIT_INTERVAL = .25

# 変数の定義 ##########

# voxelの寸法 (定数を変更したらinit_boundsを呼び出して再計算すること)
voxel_dim = [
    CAMERA_PIXEL_SIZE * BINNING / MAGNIFICATION, 
    CAMERA_PIXEL_SIZE * BINNING / MAGNIFICATION, 
    Z_SLICE_LENGTH
]

# 3D画像データの寸法 (定数を変更したらinit_boundsを呼び出して再計算すること)
cx, cy, cz = (
    0 * voxel_dim[0] * REDUCE_RATIO, 
    0 * voxel_dim[1] * REDUCE_RATIO, 
    0 * voxel_dim[2] * Z_STEP
)
bounds = (
    -cx/2, cx/2, 
    -cy/2, cy/2, 
    -cz/2, cz/2
)

# 動画のエンドポイントのリストを格納する
state_store = []

# コントロール用ウィジェットの初期化 ##########

# camera position
slider_ox = widgets.FloatSlider(step=0.1, description='x:', layout=widgets.Layout(width='33%'))
slider_oy = widgets.FloatSlider(step=0.1, description='y:', layout=widgets.Layout(width='33%'))
slider_oz = widgets.FloatSlider(step=0.1, description='z:', layout=widgets.Layout(width='33%'))
menu_axis = widgets.Dropdown(value='z', options=['x', 'y', 'z'], layout=widgets.Layout(width='5%'))
slider_ha = widgets.FloatSlider(value=0, min=0.0, max=360.0, step=1.0, description='theta:', layout=widgets.Layout(width='30%'))
slider_va = widgets.FloatSlider(value=0, min=-180, max=180.0, step=1.0, description='phi:', layout=widgets.Layout(width='30%'))
slider_d = widgets.FloatSlider(value=1000, min=1, max=3000, step=1.0, description='distance:', layout=widgets.Layout(width='30%'))

# channel contrast
check_ch, slider_ch = [], []
for ch in range(1, 5):
    check_ch.append(widgets.Checkbox(value=True, description=f'{ch}:', layout=widgets.Layout(width='25%')))
    slider_ch.append(widgets.IntRangeSlider(step=1, layout=widgets.Layout(width='70%')))

# clipping planes
slider_plane_x = widgets.IntRangeSlider(step=1, description='x slice:', layout=widgets.Layout(width='95%'))
slider_plane_y = widgets.IntRangeSlider(step=1, description='y slice:', layout=widgets.Layout(width='95%'))
slider_plane_z = widgets.IntRangeSlider(step=1, description='z slice:', layout=widgets.Layout(width='95%'))
slider_plane_zx = widgets.FloatSlider(value=0, min=-90, max=90.0, step=1, description='z_x:', layout=widgets.Layout(width='95%'))
slider_plane_zy = widgets.FloatSlider(value=0, min=-90, max=90.0, step=1, description='z_y:', layout=widgets.Layout(width='95%'))

# alpha blending
check_alpha = widgets.Checkbox(description='alpha blending')

# 3D画像データの寸法を計算する
def init_bounds(data):
    global voxel_dim, cx, cy, cz, bounds
    voxel_dim = [
        CAMERA_PIXEL_SIZE * BINNING / MAGNIFICATION, 
        CAMERA_PIXEL_SIZE * BINNING / MAGNIFICATION, 
        Z_SLICE_LENGTH
    ]
    cx, cy, cz = (
        data.shape[2] * voxel_dim[0] * REDUCE_RATIO, 
        data.shape[1] * voxel_dim[1] * REDUCE_RATIO, 
        data.shape[0] * voxel_dim[2] * Z_STEP
    )
    bounds = (
        -cx/2, cx/2, 
        -cy/2, cy/2, 
        -cz/2, cz/2
    )


# コントロール用ウィジェットの初期化処理
# objはMultiMIPオブジェクトのみ指定可能
def init_controls(plot, obj, axis='z', phi=90, distance=(1000, 1, 3000)):
    global _plot, _obj
    _plot, _obj = plot, obj
    data1 = _obj.volume_list[0]
    
    # camera position
    slider_ox.min, slider_ox.max, slider_ox.value = -cx*2, cx*2, 0
    slider_oy.min, slider_oy.max, slider_oy.value = -cy*2, cy*2, 0
    slider_oz.min, slider_oz.max, slider_oz.value = -cz*10, cz*10, 0
    menu_axis.value = axis
    slider_va.value = phi
    slider_d.min, slider_d.max, slider_d.value = distance[1], distance[2], distance[0]
    
    # channel contrast
    for i, data in enumerate(_obj.volume_list):
        min_, max_ = int(data.min()), int(data.max())
        slider_ch[i].max, slider_ch[i].min = max_, min_
        slider_ch[i].value = [min_, max_]
    
    # clipping planes
    slider_plane_x.min, slider_plane_x.max, slider_plane_x.value = 0, data1.shape[2], [0, data1.shape[2]]
    slider_plane_y.min, slider_plane_y.max, slider_plane_y.value = 0, data1.shape[1], [0, data1.shape[1]]
    slider_plane_z.min, slider_plane_z.max, slider_plane_z.value = 0, data1.shape[0], [0, data1.shape[0]]
    
    # alpha blending
    check_alpha.value=_obj.alpha_blending

# camera position change event handler
def compute_camera_pos_x(ox, oy, oz, theta, phi, d):
    phi_ = np.pi * phi / 180.0
    yz_, x_ = d * np.sin(phi_), -d * np.cos(phi_)
    theta_ = np.pi * theta / 180.0
    y_, z_ = yz_ * np.sin(-theta_), yz_ * np.cos(-theta_)
    #ox_, oy_, oz_ =  cx/2 + ox, cy/2 + oy, cz/2 + oz
    ox_, oy_, oz_ =  ox, oy, oz
    return [ox_ + x_, oy_ + y_, oz_ + z_, ox_, oy_, oz_, -np.sin(phi_), -np.cos(phi_) * np.sin(-theta_), -np.cos(phi_) * np.cos(-theta_)]

def compute_camera_pos_y(ox, oy, oz, theta, phi, d):
    phi_ = np.pi * phi / 180.0
    zx_, y_ = d * np.sin(phi_), -d * np.cos(phi_)
    theta_ = np.pi * theta / 180.0
    z_, x_ = zx_ * np.sin(-theta_), zx_ * np.cos(-theta_)
    #ox_, oy_, oz_ =  cx/2 + ox, cy/2 + oy, cz/2 + oz
    ox_, oy_, oz_ =  ox, oy, oz
    return [ox_ + x_, oy_ + y_, oz_ + z_, ox_, oy_, oz_, -np.cos(phi_) * np.cos(-theta_), -np.sin(phi_), -np.cos(phi_) * np.sin(-theta_)]

def compute_camera_pos_z(ox, oy, oz, theta, phi, d):
    phi_ = np.pi * phi / 180.0
    xy_, z_ = d * np.sin(phi_), -d * np.cos(phi_)
    theta_ = np.pi * theta / 180.0
    x_, y_ = xy_ * np.sin(-theta_), xy_ * np.cos(-theta_)
    #ox_, oy_, oz_ =  cx/2 + ox, cy/2 + oy, cz/2 + oz
    ox_, oy_, oz_ =  ox, oy, oz
    return [ox_ + x_, oy_ + y_, oz_ + z_, ox_, oy_, oz_, -np.cos(phi_) * np.sin(-theta_), -np.cos(phi_) * np.cos(-theta_), -np.sin(phi_)]

compute_camera_pos_table = {
    'x': compute_camera_pos_x, 
    'y': compute_camera_pos_y, 
    'z': compute_camera_pos_z, 
}

def compute_camera_pos(ox, oy, oz, axis, theta, phi, d):
    return compute_camera_pos_table[axis](ox, oy, oz, theta, phi, d)

def update_camera_pos(*change):
    _plot.camera = compute_camera_pos(
        slider_ox.value, 
        slider_oy.value, 
        slider_oz.value, 
        menu_axis.value, 
        slider_ha.value, 
        slider_va.value, 
        slider_d.value, 
    )

# channel contrast change event handler
def update_color_range_list(*change):
    color_range_list = []
    for i, data in enumerate(_obj.volume_list):
        color_range_list.append([slider_ch[i].value[0], slider_ch[i].value[1]])
    _obj.color_range_list = color_range_list

# channel opacity change event handler
def update_opacity_function_list(*change):
    opacity_function_list = []
    for i, data in enumerate(_obj.volume_list):
        opacity_function_list.append([0.0, 0.0, 1.0, 1.0] if check_ch[i].value else [0.0, 0.0, 1.0, 0.0])
    _obj.opacity_function_list = opacity_function_list

# clipping planes change event handler
def update_plane(*change):
    zx_angle = np.pi * slider_plane_zx.value / 180.0
    zy_angle = np.pi * slider_plane_zy.value / 180.0
    nv_zx = -np.sin(zx_angle) * np.cos(zy_angle)
    nv_zy = -np.cos(zx_angle) * np.sin(zy_angle)
    nv_zz = np.cos(zx_angle) * np.cos(zy_angle)
    _plot.clipping_planes=[
        [1, 0, 0, -slider_plane_x.value[0] * voxel_dim[0] * REDUCE_RATIO + cx/2],
        [0, 1, 0, -slider_plane_y.value[0] * voxel_dim[1] * REDUCE_RATIO + cy/2],
        [nv_zx, nv_zy, nv_zz, -slider_plane_z.value[0] * voxel_dim[2] * Z_STEP + cz/2],
        [-1, 0, 0, slider_plane_x.value[1] * voxel_dim[0] * REDUCE_RATIO - cx/2],
        [0, -1, 0, slider_plane_y.value[1] * voxel_dim[1] * REDUCE_RATIO - cy/2],
#        [-nv_zx, -nv_zy, -nv_zz, slider_plane_z.value[1] * voxel_dim[2] * Z_STEP - cz/2],
        [0, 0, -1, slider_plane_z.value[1] * voxel_dim[2] * Z_STEP - cz/2],
    ]

# alpha blending change event handler
def update_plane_alpha(*change):
    _obj.alpha_blending = check_alpha.value

# refresh view
def refresh():
    update_camera_pos()
    update_color_range_list()
    update_opacity_function_list()
    update_plane()
    update_plane_alpha()

# observe events
def observe_control_events():
    slider_ox.observe(update_camera_pos, names='value')
    slider_oy.observe(update_camera_pos, names='value')
    slider_oz.observe(update_camera_pos, names='value')
    menu_axis.observe(update_camera_pos, names='value')
    slider_ha.observe(update_camera_pos, names='value')
    slider_va.observe(update_camera_pos, names='value')
    slider_d.observe(update_camera_pos, names='value')    
    for i, j in zip(slider_ch, check_ch):
        i.observe(update_color_range_list, names='value')
        j.observe(update_opacity_function_list, names='value')
    slider_plane_x.observe(update_plane, names='value')
    slider_plane_y.observe(update_plane, names='value')
    slider_plane_z.observe(update_plane, names='value')
    slider_plane_zx.observe(update_plane, names='value')
    slider_plane_zy.observe(update_plane, names='value')
    check_alpha.observe(update_plane_alpha, names='value')

# unobserve events
def unobserve_control_events():
    slider_ox.unobserve(update_camera_pos, names='value')
    slider_oy.unobserve(update_camera_pos, names='value')
    slider_oz.unobserve(update_camera_pos, names='value')
    menu_axis.unobserve(update_camera_pos, names='value')
    slider_ha.unobserve(update_camera_pos, names='value')
    slider_va.unobserve(update_camera_pos, names='value')
    slider_d.unobserve(update_camera_pos, names='value')
    for i, j in zip(slider_ch, check_ch):
        i.unobserve(update_color_range_list, names='value')
        j.unobserve(update_opacity_function_list, names='value')
    slider_plane_x.unobserve(update_plane, names='value')
    slider_plane_y.unobserve(update_plane, names='value')
    slider_plane_z.unobserve(update_plane, names='value')
    slider_plane_zx.unobserve(update_plane, names='value')
    slider_plane_zy.unobserve(update_plane, names='value')
    check_alpha.unobserve(update_plane_alpha, names='value')

observe_control_events()


# ウィジェットの表示 ##########
def display_controls():
    display(
        widgets.VBox([
            widgets.HBox([slider_ox, slider_oy, slider_oz]),
            widgets.HBox([menu_axis, slider_ha, slider_va, slider_d]),
            widgets.HBox([
                widgets.VBox([
                    widgets.HBox([i, j]) for i, j, k in zip(check_ch, slider_ch, _obj.volume_list)
                ], layout=widgets.Layout(width='55%')),
                widgets.VBox([
                    slider_plane_x, 
                    slider_plane_y, 
                    slider_plane_z, 
                    slider_plane_zx, 
                    slider_plane_zy, 
                    check_alpha, 
                ], layout=widgets.Layout(width='45%')),
            ]),
        ])
    )
    refresh()


# 動画作成用ウィジェットの初期化 ##########

input_filename = widgets.Text(value='sequence.json', description='file name:')
button_load = widgets.Button(description='load sequence')
button_save = widgets.Button(description='save sequence')
button_reset = widgets.Button(description='init sequence')
input_state_duration = widgets.BoundedFloatText(value=6, min=0, max=60, step=.1, description='duration:', layout=widgets.Layout(width='15%'))
button_store = widgets.Button(description='add to sequence')
input_state_index = widgets.BoundedIntText(value=0, min=0, max=99, step=1, description='No.:', layout=widgets.Layout(width='15%'))
button_seek = widgets.Button(description='show')
button_change = widgets.Button(description='change')
button_insert = widgets.Button(description='insert')
button_remove = widgets.Button(description='remove')

output_state = widgets.Output(layout=widgets.Layout(width='95%'))

input_dry_run_fps = widgets.BoundedIntText(value=5, min=1, max=30, step=1, description='fps:', layout=widgets.Layout(width='15%'))
input_dry_run_delay = widgets.BoundedIntText(value=3, min=0, max=60, step=1, description='start delay:', layout=widgets.Layout(width='15%'))
button_dry_run = widgets.Button(description='dry run')
label_dry_run = widgets.Label(value='')
button_camera_test = widgets.Button(description='camera test (experimental)')

input_movie_fps = widgets.BoundedIntText(value=30, min=1, max=60, step=1, description='fps:', layout=widgets.Layout(width='15%'))
input_movie_filename = widgets.Text(value='hoge.mp4', description='file name:')
button_movie = widgets.Button(description='generate movie')
progress_movie = widgets.IntProgress(value=0, min=0, max=100, orientation='horizontal')
label_movie = widgets.Label(value='')

def print_state():
    output_state.clear_output()
    with output_state:
        display(pd.DataFrame(state_store))

def on_store(*b):
    state_, last_state = get_state(), state_store[-1].copy()
    if len(state_store) < 100:
        state_store.append(get_state())
        print_state()

def on_change(*b):
    index = input_state_index.value
    if index >= len(state_store):
        return
    state_ = get_state()
    if index == 0:
        state_['duration'] = 0
    state_store[index] = state_
    print_state()

def on_insert(*b):
    index = input_state_index.value
    if index >= len(state_store):
        return
    state_ = get_state()
    if index == 0:
        state_['duration'] = 0
    state_store.insert(index, state_)
    print_state()

def on_save(*b):
    with open(input_filename.value, 'w') as file:
        json.dump(state_store, file)
    print_state()

def on_load(*b):
    global state_store
    with open(input_filename.value, 'r') as file:
        state_store = json.load(file)
    print_state()

def on_reset(*b):
    state = get_state(0)
    global state_store
    state_store = [state]
    print_state()

def on_seek(*b):
    if input_state_index.value < len(state_store):
        _ = state_store[input_state_index.value]
        if _['duration']:
            input_state_duration.value = _['duration']
        update_movie_camera_pos(
            _['camera_pos'], 
            _['ox'], 
            _['oy'], 
            _['oz'], 
            _['axis'], 
            _['ha'], 
            _['va'], 
            _['d']
        )
        update_movie_color_range_list(_['slider_ch'])
        update_movie_opacity_function_list(_['check_ch'])
        update_movie_plane(_['plane_x'], _['plane_y'], _['plane_z'])
        update_movie_alpha_blending(_['alpha'])

def on_remove(*b):
    if 1 < len(state_store) and input_state_index.value < len(state_store):
        del state_store[input_state_index.value]
        if len(state_store) == 1:
            state_store[0]['duration'] = 0
        print_state()

def on_dry_run(*b):
    print_state()
    dry_run(input_dry_run_fps.value)

auto_play = False

def on_camera_test(*b):
    print_state()
    global auto_play
    if auto_play:
        auto_play = False
        _plot.stop_auto_play()
    else:
        auto_play = True
        camera_test(input_dry_run_fps.value)
        _plot.start_auto_play()

movie_thread = None
def on_movie(*b):
    global movie_thread
    if movie_thread:
        global sequence_stop
        sequence_stop = True
        movie_thread = None
        button_movie.description = 'generate movie'
    else:
        print_state()
        movie_thread = threading.Thread(
            target=generate_movie, 
            args=(input_movie_filename.value, input_movie_fps.value)
        )
        movie_thread.start()
        button_movie.description = 'stop'

button_store.on_click(on_store)
button_seek.on_click(on_seek)
button_change.on_click(on_change)
button_insert.on_click(on_insert)
button_remove.on_click(on_remove)
button_dry_run.on_click(on_dry_run)
button_camera_test.on_click(on_camera_test)
button_movie.on_click(on_movie)
button_reset.on_click(on_reset)
button_load.on_click(on_load)
button_save.on_click(on_save)
print_state()

def get_state(duration=None):
    if duration is None:
        duration = input_state_duration.value
    camera_pos = compute_camera_pos(
        slider_ox.value, 
        slider_oy.value, 
        slider_oz.value, 
        menu_axis.value, 
        slider_ha.value, 
        slider_va.value, 
        slider_d.value, 
    )
    return dict(
        duration = duration, 
        camera_pos = camera_pos, 
        ox = slider_ox.value,
        oy = slider_oy.value, 
        oz = slider_oz.value, 
        axis = menu_axis.value, 
        ha = slider_ha.value, 
        va = slider_va.value, 
        d = slider_d.value, 
        check_ch = [x.value for x, _ in zip(check_ch, _obj.volume_list)], 
        slider_ch = [x.value for x, _ in zip(slider_ch, _obj.volume_list)], 
        plane_x = slider_plane_x.value, 
        plane_y = slider_plane_y.value, 
        plane_z = slider_plane_z.value, 
        alpha = check_alpha.value
    )


# ウィジェットの表示 ##########
def display_movie_controls():
    display(
        widgets.VBox([
            widgets.HBox([
                button_reset,
                input_filename, 
                button_load,
                button_save,
            ]),
            widgets.HBox([
                input_state_duration, 
                button_store,
                input_state_index,
                button_seek, 
                button_change, 
                button_insert, 
                button_remove, 
            ]),
            widgets.HBox([
                input_dry_run_fps, 
                input_dry_run_delay, 
                button_dry_run, 
                label_dry_run, 
                #button_camera_test, 
            ]),
            widgets.HBox([
                input_movie_fps, 
                input_movie_filename,
                button_movie,
                progress_movie,
                label_movie, 
            ]),
            output_state, 
        ])
    )
    print_state()



# 動画作成処理 ##########

def get_movie_camera_pos(camera_pos, ox, oy, oz, axis, theta, phi, d):
    unobserve_control_events()
    slider_ox.value = ox
    slider_oy.value = oy
    slider_oz.value = oz
    menu_axis.value = axis
    slider_ha.value = theta
    slider_va.value = phi
    slider_d.value = d
    observe_control_events()
    if camera_pos is None or all(camera_pos == np.array((0,0,0,0,0,0,0,0,0))):
        return compute_camera_pos(ox, oy, oz, axis, theta, phi, d)
    else:
        return camera_pos

def update_movie_camera_pos(camera_pos, ox, oy, oz, axis, theta, phi, d):
    _plot.camera = get_movie_camera_pos(camera_pos, ox, oy, oz, axis, theta, phi, d)

def update_movie_color_range_list(slider_ch_):
    unobserve_control_events()
    for i, j in zip(slider_ch, slider_ch_):
        i.value = j
    observe_control_events()
    update_color_range_list()

def update_movie_opacity_function_list(check_ch_):
    unobserve_control_events()
    for i, j in zip(check_ch, check_ch_):
        i.value = j
    observe_control_events()
    update_opacity_function_list()

def update_movie_plane(plane_x, plane_y, plane_z):
    unobserve_control_events()
    slider_plane_x.value = plane_x
    slider_plane_y.value = plane_y
    slider_plane_z.value = plane_z
    observe_control_events()
    _plot.clipping_planes=[
        [1, 0, 0, -plane_x[0] * voxel_dim[0] * REDUCE_RATIO],
        [0, 1, 0, -plane_y[0] * voxel_dim[1] * REDUCE_RATIO],
        [0, 0, 1, -plane_z[0] * voxel_dim[2] * Z_STEP],
        [-1, 0, 0, plane_x[1] * voxel_dim[0] * REDUCE_RATIO],
        [0, -1, 0, plane_y[1] * voxel_dim[1] * REDUCE_RATIO],
        [0, 0, -1, plane_z[1] * voxel_dim[2] * Z_STEP],
    ]

def update_movie_alpha_blending(alpha):
    unobserve_control_events()
    check_alpha.value = alpha
    observe_control_events()
    _obj.alpha_blending = alpha

def sequence_movie(fps):
    for i in range(len(state_store) - 1):
        state0 = state_store[i]
        state1 = state_store[i + 1]
        c = int(state1['duration'] * fps + 1)
        camera_pos_ = np.linspace(state0['camera_pos'], state1['camera_pos'], c)
        ox_ = np.linspace(state0['ox'], state1['ox'], c)
        oy_ = np.linspace(state0['oy'], state1['oy'], c)
        oz_ = np.linspace(state0['oz'], state1['oz'], c)
        axis_ = state1['axis']
        axis_chg = state0['axis'] != axis_
        ha_ = np.linspace(state0['ha'], state1['ha'], c)
        va_ = np.linspace(state0['va'], state1['va'], c)
        d_  = np.linspace(state0['d'] , state1['d'], c)
        slider_ch_ = [np.linspace(state0_, state1_, c) + .5 for state0_, state1_ in zip(state0['slider_ch'], state1['slider_ch'])]
        plane_x_ = np.linspace(state0['plane_x'], state1['plane_x'], c)+.5
        plane_y_ = np.linspace(state0['plane_y'], state1['plane_y'], c)+.5
        plane_z_ = np.linspace(state0['plane_z'], state1['plane_z'], c)+.5
        camera_pos, color_range_list, opacity_function_list, plane, alpha_blending = None, None, None, None, None
        for j in range(1 if i > 0 else 0, c):
            camera_pos__ = (camera_pos_[j] if axis_chg else None, ox_[j], oy_[j], oz_[j], axis_, ha_[j], va_[j], d_[j])
            color_range_list__ = ([(ch[j][0], ch[j][1]) for ch in slider_ch_],)
            opacity_function_list__ = (state1['check_ch'],)
            plane__ = ((plane_x_[j][0], plane_x_[j][1]), (plane_y_[j][0], plane_y_[j][1]), (plane_z_[j][0], plane_z_[j][1]))
            alpha_blending__ = state0['alpha'] if j < c - 1 else state1['alpha']
            retval = dict(index=i+1)
            if camera_pos != camera_pos__:
                retval['camera_pos'] = camera_pos = camera_pos__
            if color_range_list != color_range_list__:
                retval['color_range_list'] = color_range_list = color_range_list__
            if opacity_function_list != opacity_function_list__:
                retval['opacity_function_list'] = opacity_function_list = opacity_function_list__
            if plane != plane__:
                retval['plane'] = plane = plane__
            if alpha_blending != alpha_blending__:
                retval['alpha_blending'] = alpha_blending = alpha_blending__
            yield retval

def dry_run(fps=5):
    for i in range(input_dry_run_delay.value, 0, -1):
        label_dry_run.value = f'count down: {i}'
        time.sleep(1)
    for seq in sequence_movie(fps):
        label_dry_run.value = '{} -> {}'.format(seq['index'] - 1, seq['index'])
        if 'camera_pos' in seq:
            update_movie_camera_pos(*seq['camera_pos'])
        if 'color_range_list' in seq:
            update_movie_color_range_list(*seq['color_range_list'])
        if 'opacity_function_list' in seq:
            update_movie_opacity_function_list(*seq['opacity_function_list'])
        if 'plane' in seq:
            update_movie_plane(*seq['plane'])
        if 'alpha_blending' in seq:
            update_movie_alpha_blending(seq['alpha_blending'])
        time.sleep(1/fps)
    label_dry_run.value = ''

def camera_test(fps=5):
    _plot.fps = fps
    t = 0.0
    ca = {}
    for seq in sequence_movie(_plot.fps):
        ca[str(t)] = get_movie_camera_pos(*seq['camera_pos'])
        t += 1.0 / _plot.fps
    _plot.camera_animation = ca

def generate_movie(movie_filename, fps, bitrate='8192k'):
    # H.264ライセンス問題: https://av.watch.impress.co.jp/docs/20031118/mpegla.htm
    duration_list = [i['duration'] for i in state_store[1:]]
    if sum(duration_list) > 12 * 60:
        with output_state:
            print('H.264 movie should not be greater than 12 minutes.')
        return

    with output_state:
        print('generating movie started at ' + str(datetime.datetime.now()))
    h = _plot.height * 2
    w = int(h * 16/9) & ~0x0f
    for seq_number, seq in enumerate(sequence_movie(fps)):
        pass
    progress_movie.value = 0
    progress_movie.max = seq_number + 1
    with FFmpegFrameWriter(movie_filename, fps=fps, size=(w, h), bitrate=bitrate) as writer:
        global sequence_stop
        sequence_stop = False
        for i, seq in enumerate(sequence_movie(fps)):
            try:
                progress_movie.value = i + 1
                label_movie.value = '{} -> {}'.format(seq['index'] - 1, seq['index'])
                if 'camera_pos' in seq:
                    update_movie_camera_pos(*seq['camera_pos']); time.sleep(MOVIE_WAIT_INTERVAL)
                if 'color_range_list' in seq:
                    update_movie_color_range_list(*seq['color_range_list']); time.sleep(MOVIE_WAIT_INTERVAL)
                if 'opacity_function_list' in seq:
                    update_movie_opacity_function_list(*seq['opacity_function_list']); time.sleep(MOVIE_WAIT_INTERVAL)
                if 'plane' in seq:
                    update_movie_plane(*seq['plane']); time.sleep(MOVIE_WAIT_INTERVAL)
                if 'alpha_blending' in seq:
                    update_movie_alpha_blending(seq['alpha_blending']); time.sleep(MOVIE_WAIT_INTERVAL)
                _plot.screenshot = ''
                _plot.fetch_screenshot(only_canvas=False)
                while not (_plot.screenshot or sequence_stop):
                    time.sleep(MOVIE_WAIT_INTERVAL)
                if sequence_stop:
                    with output_state:
                        print('generating movie stopped at ' + str(datetime.datetime.now()))
                    break
                img = np.asarray(Image.open(io.BytesIO(base64.b64decode(_plot.screenshot))))
                if img.shape[1] > w:
                    offset = (img.shape[1] - w) // 2
                    writer.add(img[:, offset:(offset+w), :3])
                else:
                    offset = (w - img.shape[1]) // 2
                    writer.frame[:, offset:(offset+img.shape[1]), :] = img[:, :, :3]
                    writer.add_frame()
            except Exception as excep:
                with output_state:
                    print(excep)
        else:
            with output_state:
                print('generating movie finished at ' + str(datetime.datetime.now()))
            on_movie()
        label_movie.value = ''
