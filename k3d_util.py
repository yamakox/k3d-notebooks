import numpy as np
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from scipy import interpolate
from frame_writer import *
import base64, io, time

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

# 最大4色まで対応する(2番目の要素からR, G, B)
COLOR_MAP_LIST = [
    [(0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)], 
    [(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 1.0, 0.0)],
    [(0.0, 0.8, 0.6, 0.0), (1.0, 0.8, 0.6, 0.0)],
    [(0.0, 1.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)],
]


# 変数の定義 ##########

# voxelの寸法 (定数を変更したらinit_boundsを実行して再計算すること)
voxel_dim = [
    CAMERA_PIXEL_SIZE * BINNING / MAGNIFICATION, 
    CAMERA_PIXEL_SIZE * BINNING / MAGNIFICATION, 
    Z_SLICE_LENGTH
]

# 3D画像データの寸法 (定数を変更したらinit_boundsを実行して再計算すること)
cx, cy, cz = (
    0 * voxel_dim[0] * REDUCE_RATIO, 
    0 * voxel_dim[1] * REDUCE_RATIO, 
    0 * voxel_dim[2] * Z_STEP
)

# 動画のエンドポイントのリストを格納する
state_store = []

# コントロール用ウィジェットの初期化 ##########

# camera position
slider_ox = widgets.FloatSlider(step=0.1, description='x:', layout=widgets.Layout(width='33%'))
slider_oy = widgets.FloatSlider(step=0.1, description='y:', layout=widgets.Layout(width='33%'))
slider_oz = widgets.FloatSlider(step=0.1, description='z:', layout=widgets.Layout(width='33%'))
slider_ha = widgets.FloatSlider(value=0, min=0.0, max=360.0, step=1.0, description='theta:', layout=widgets.Layout(width='33%'))
slider_va = widgets.FloatSlider(value=45, min=-89.0, max=89.0, step=1.0, description='phi:', layout=widgets.Layout(width='33%'))
slider_d = widgets.FloatSlider(value=1000, min=1, max=3000, step=1.0, description='distance:', layout=widgets.Layout(width='33%'))

# channel contrast
check_ch, slider_ch = [], []
for ch in range(1, 5):
    check_ch.append(widgets.Checkbox(value=True, description=f'{ch}:', layout=widgets.Layout(width='25%')))
    slider_ch.append(widgets.IntRangeSlider(step=1, layout=widgets.Layout(width='70%')))

# clipping planes
slider_plane_x = widgets.IntRangeSlider(step=1, description='x slice:', layout=widgets.Layout(width='95%'))
slider_plane_y = widgets.IntRangeSlider(step=1, description='y slice:', layout=widgets.Layout(width='95%'))
slider_plane_z = widgets.IntRangeSlider(step=1, description='z slice:', layout=widgets.Layout(width='95%'))

# alpha blending
check_alpha = widgets.Checkbox(description='alpha blending')

# 3D画像データの寸法を計算する
def init_bounds(data):
    global voxel_dim, cx, cy, cz
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

# コントロール用ウィジェットの初期化処理
def init_controls(plot, obj, phi=45, distance=(1000, 1, 3000)):
    global _plot, _obj
    _plot, _obj = plot, obj
    data1 = _obj.volume_list[0]
    
    # camera position
    slider_ox.min, slider_ox.max, slider_ox.value = -cx*2, cx*2, 0
    slider_oy.min, slider_oy.max, slider_oy.value = -cy*2, cy*2, 0
    slider_oz.min, slider_oz.max, slider_oz.value = -cz*10, cz*10, 0
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
def update_camera_pos(*change):
    phi = np.pi * slider_va.value / 180.0
    xy, z = slider_d.value * np.cos(phi), slider_d.value * np.sin(phi)
    theta = np.pi * slider_ha.value / 180.0
    x, y = xy * np.sin(theta), xy * np.cos(theta)
    ox, oy, oz =  cx/2 + slider_ox.value, cy/2 + slider_oy.value, cz/2 + slider_oz.value
    _plot.camera = [ox - x, oy - y, oz + z, ox, oy, oz, 0, 0, 1]

# channel contrast change event handler
def update_color_range_list(*change):
    color_range_list = []
    for i, data in enumerate(_obj.volume_list):
        color_range_list.append([slider_ch[i].value[0], slider_ch[i].value[1] if check_ch[i].value else 10000000])
    _obj.color_range_list = color_range_list

# clipping planes change event handler
def update_plane(*change):
    _plot.clipping_planes=[
        [1, 0, 0, -slider_plane_x.value[0] * voxel_dim[0] * REDUCE_RATIO],
        [0, 1, 0, -slider_plane_y.value[0] * voxel_dim[1] * REDUCE_RATIO],
        [0, 0, 1, -slider_plane_z.value[0] * voxel_dim[2] * Z_STEP],
        [-1, 0, 0, slider_plane_x.value[1] * voxel_dim[0] * REDUCE_RATIO],
        [0, -1, 0, slider_plane_y.value[1] * voxel_dim[1] * REDUCE_RATIO],
        [0, 0, -1, slider_plane_z.value[1] * voxel_dim[2] * Z_STEP],
    ]

# alpha blending change event handler
def update_plane_alpha(*change):
    _obj.alpha_blending = check_alpha.value

# refresh view
def refresh():
    update_camera_pos()
    update_color_range_list()
    update_plane()
    update_plane_alpha()

# observe events
def observe_control_events():
    slider_ox.observe(update_camera_pos, names='value')
    slider_oy.observe(update_camera_pos, names='value')
    slider_oz.observe(update_camera_pos, names='value')
    slider_ha.observe(update_camera_pos, names='value')
    slider_va.observe(update_camera_pos, names='value')
    slider_d.observe(update_camera_pos, names='value')    
    for i, j in zip(check_ch, slider_ch):
        i.observe(update_color_range_list, names='value')
        j.observe(update_color_range_list, names='value')
    slider_plane_x.observe(update_plane, names='value')
    slider_plane_y.observe(update_plane, names='value')
    slider_plane_z.observe(update_plane, names='value')
    check_alpha.observe(update_plane_alpha, names='value')

# unobserve events
def unobserve_control_events():
    slider_ox.unobserve(update_camera_pos, names='value')
    slider_oy.unobserve(update_camera_pos, names='value')
    slider_oz.unobserve(update_camera_pos, names='value')
    slider_ha.unobserve(update_camera_pos, names='value')
    slider_va.unobserve(update_camera_pos, names='value')
    slider_d.unobserve(update_camera_pos, names='value')
    for i, j in zip(check_ch, slider_ch):
        i.unobserve(update_color_range_list, names='value')
        j.unobserve(update_color_range_list, names='value')
    slider_plane_x.unobserve(update_plane, names='value')
    slider_plane_y.unobserve(update_plane, names='value')
    slider_plane_z.unobserve(update_plane, names='value')
    check_alpha.unobserve(update_plane_alpha, names='value')

observe_control_events()


# ウィジェットの表示 ##########
def display_controls():
    display(
        widgets.VBox([
            widgets.HBox([slider_ox, slider_oy, slider_oz]),
            widgets.HBox([slider_ha, slider_va, slider_d]),
            widgets.HBox([
                widgets.VBox([
                    widgets.HBox([i, j]) for i, j, k in zip(check_ch, slider_ch, _obj.volume_list)
                ], layout=widgets.Layout(width='55%')),
                widgets.VBox([
                    slider_plane_x, 
                    slider_plane_y, 
                    slider_plane_z, 
                    check_alpha, 
                ], layout=widgets.Layout(width='45%')),
            ]),
        ])
    )
    refresh()


# 動画作成用ウィジェットの初期化 ##########

button_reset = widgets.Button(description='init sequences')
input_state_duration = widgets.BoundedIntText(value=6, min=1, max=60, step=1, description='duration:', layout=widgets.Layout(width='15%'))
button_store = widgets.Button(description='add sequence')
input_state_index = widgets.BoundedIntText(value=0, min=0, max=99, step=1, description='No.:', layout=widgets.Layout(width='15%'))
button_seek = widgets.Button(description='show')
button_change = widgets.Button(description='change')
button_remove = widgets.Button(description='remove')

output_state = widgets.Output(layout=widgets.Layout(width='95%'))

input_dry_run_fps = widgets.BoundedIntText(value=5, min=1, max=10, step=1, description='fps:', layout=widgets.Layout(width='15%'))
button_dry_run = widgets.Button(description='dry run')

input_movie_fps = widgets.BoundedIntText(value=30, min=1, max=60, step=1, description='fps:', layout=widgets.Layout(width='15%'))
input_movie_filename = widgets.Text(value='hoge.mp4', description='file name:')
button_movie = widgets.Button(description='generate movie')
button_movie_stop = widgets.Button(description='stop')

def print_state():
    output_state.clear_output()
    with output_state:
        display(pd.DataFrame(state_store))

def on_store(*b):
    state_, last_state = get_state(), state_store[-1].copy()
    del state_['duration'], last_state['duration']
    if state_ != last_state and len(state_store) < 100:
        state_store.append(get_state())
        print_state()

def on_change(*b):
    state_ = get_state()
    del state_['duration']
    index = input_state_index.value
    if index >= len(state_store):
        return
    if index > 0:
        prev_state = state_store[index - 1].copy()
        del prev_state['duration']
        if state_ == prev_state:
            return
    if index <len(state_store) - 1:
        next_state = state_store[index + 1].copy()
        del next_state['duration']
        if state_ == next_state:
            return
    state_ = get_state()
    if index == 0:
        state_['duration'] = 0
    state_store[index] = state_
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
        update_movie_camera_pos(_['ox'], _['oy'], _['oz'], _['ha'], _['va'], _['d'])
        update_movie_color_range_list(_['slider_ch'], _['check_ch'])
        update_movie_plane(_['plane_x'], _['plane_y'], _['plane_z'])
        update_movie_alpha_blending(_['alpha'])

def on_remove(*b):
    if 1 < len(state_store) and input_state_index.value < len(state_store):
        del state_store[input_state_index.value]
        if len(state_store) == 1:
            state_store[0]['duration'] = 0
        print_state()

def on_dry_run(*b):
    dry_run(input_dry_run_fps.value)

def on_movie(*b):
    generate_movie(input_movie_filename.value, input_movie_fps.value)

def on_movie_stop(*b):
    global sequence_stop
    sequence_stop = True

button_store.on_click(on_store)
button_reset.on_click(on_reset)
button_seek.on_click(on_seek)
button_change.on_click(on_change)
button_remove.on_click(on_remove)
button_dry_run.on_click(on_dry_run)
button_movie.on_click(on_movie)
button_movie_stop.on_click(on_movie_stop)
print_state()

def get_state(duration=None):
    if duration is None:
        duration = input_state_duration.value
    return dict(
        duration = duration, 
        ox = slider_ox.value,
        oy = slider_oy.value, 
        oz = slider_oz.value, 
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
                input_state_duration, 
                button_store,
                input_state_index,
                button_seek, 
                button_change, 
                button_remove, 
                button_reset,
            ]),
            widgets.HBox([
                input_dry_run_fps, 
                button_dry_run, 
            ]),
            widgets.HBox([
                input_movie_fps, 
                input_movie_filename,
                button_movie,
                button_movie_stop,
            ]),
            output_state, 
        ])
    )
    on_reset()



# 動画作成処理 ##########

def update_movie_camera_pos(ox, oy, oz, theta, phi, d):
    unobserve_control_events()
    slider_ox.value = ox
    slider_oy.value = oy
    slider_oz.value = oz
    slider_ha.value = theta
    slider_va.value = phi
    slider_d.value = d
    observe_control_events()
    phi_ = np.pi * phi / 180.0
    xy_, z_ = d * np.cos(phi_), d * np.sin(phi_)
    theta_ = np.pi * theta / 180.0
    x_, y_ = xy_ * np.sin(theta_), xy_ * np.cos(theta_)
    ox_, oy_, oz_ =  cx/2 + ox, cy/2 + oy, cz/2 + oz
    _plot.camera = [ox_ - x_, oy_ - y_, oz_ + z_, ox_, oy_, oz_, 0, 0, 1]

def update_movie_color_range_list(slider_ch_, check_ch_):
    unobserve_control_events()
    for i, j in zip(slider_ch, slider_ch_):
        i.value = j
    for i, j in zip(check_ch, check_ch_):
        i.value = j
    observe_control_events()
    color_range_list = []
    for i, j in zip(slider_ch_, check_ch_):
        color_range_list.append([i[0], i[1] if j else 10000000])
    _obj.color_range_list = color_range_list

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
        ox_ = np.linspace(state0['ox'], state1['ox'], c)
        oy_ = np.linspace(state0['oy'], state1['oy'], c)
        oz_ = np.linspace(state0['oz'], state1['oy'], c)
        ha_ = np.linspace(state0['ha'], state1['ha'], c)
        va_ = np.linspace(state0['va'], state1['va'], c)
        d_  = np.linspace(state0['d'] , state1['d'], c)
        slider_ch_ = [(np.linspace(_0[0], _1[0], c)+.5, np.linspace(_0[1], _1[1], c)+.5) for _0, _1 in zip(state0['slider_ch'], state1['slider_ch'])]
        plane_x_ = (np.linspace(state0['plane_x'][0], state1['plane_x'][0], c)+.5, np.linspace(state0['plane_x'][1], state1['plane_x'][1], c)+.5)
        plane_y_ = (np.linspace(state0['plane_y'][0], state1['plane_y'][0], c)+.5, np.linspace(state0['plane_y'][1], state1['plane_y'][1], c)+.5)
        plane_z_ = (np.linspace(state0['plane_z'][0], state1['plane_z'][0], c)+.5, np.linspace(state0['plane_z'][1], state1['plane_z'][1], c)+.5)
        for j in range(1 if i > 0 else 0, c):
            yield dict(
                camera_pos = (ox_[j], oy_[j], oz_[j], ha_[j], va_[j], d_[j]), 
                color_range_list = ([(ch[0][j], ch[1][j]) for ch in slider_ch_], state0['check_ch'] if j < c -1 else state1['check_ch']), 
                plane = ((plane_x_[0][j], plane_x_[1][j]), (plane_y_[0][j], plane_y_[1][j]), (plane_z_[0][j], plane_z_[1][j])), 
                alpha_blending = state0['alpha'] if j < c - 1 else state1['alpha']
            )

def _interpolate(xnew, x, y):
    f = interpolate.splrep(x, y, s=0)
    return interpolate.splev(xnew, f, der=0)

# sequence_movie_splineはグニャグニャしすぎて酔いそうになるため没。
def sequence_movie_spline(fps):
    if len(state_store) < 4:    # スプライン補間のために4点必要
        return
    time_series, total = [], 0
    for i in state_store:
        total += i['duration']
        time_series.append(total)
    tp = np.linspace(0, total, total * fps + 1)
    df = pd.DataFrame(state_store)
    ox_ = _interpolate(tp, time_series, df['ox'])
    oy_ = _interpolate(tp, time_series, df['oy'])
    oz_ = _interpolate(tp, time_series, df['oz'])
    ha_ = _interpolate(tp, time_series, df['ha'])
    va_ = _interpolate(tp, time_series, df['va'])
    d_  = _interpolate(tp, time_series, df['d'])
    t = 0
    for i in range(len(state_store) - 1):
        state0 = state_store[i]
        state1 = state_store[i + 1]
        c = int(state1['duration'] * fps + 1)
        slider_ch_ = [(np.linspace(_0[0], _1[0], c)+.5, np.linspace(_0[1], _1[1], c)+.5) for _0, _1 in zip(state0['slider_ch'], state1['slider_ch'])]
        plane_x_ = (np.linspace(state0['plane_x'][0], state1['plane_x'][0], c)+.5, np.linspace(state0['plane_x'][1], state1['plane_x'][1], c)+.5)
        plane_y_ = (np.linspace(state0['plane_y'][0], state1['plane_y'][0], c)+.5, np.linspace(state0['plane_y'][1], state1['plane_y'][1], c)+.5)
        plane_z_ = (np.linspace(state0['plane_z'][0], state1['plane_z'][0], c)+.5, np.linspace(state0['plane_z'][1], state1['plane_z'][1], c)+.5)
        for j in range(1 if i > 0 else 0, c):
            yield dict(
                camera_pos = (ox_[t], oy_[t], oz_[t], ha_[t], va_[t], d_[t]), 
                color_range_list = ([(ch[0][j], ch[1][j]) for ch in slider_ch_], state0['check_ch'] if j < c -1 else state1['check_ch']), 
                plane = ((plane_x_[0][j], plane_x_[1][j]), (plane_y_[0][j], plane_y_[1][j]), (plane_z_[0][j], plane_z_[1][j])), 
                alpha_blending = state0['alpha'] if j < c - 1 else state1['alpha']
            )
            t += 1

def dry_run(fps=5):
    for seq in sequence_movie(fps):
        update_movie_camera_pos(*seq['camera_pos'])
        update_movie_color_range_list(*seq['color_range_list'])
        update_movie_plane(*seq['plane'])
        update_movie_alpha_blending(seq['alpha_blending'])
        time.sleep(1/fps)

# 致命的な問題: 前のフレームと変化がないときはscreenshot = yieldで固まってしまう。(ipywidgetsのobserve関数で変数の変化を検出している)
def generate_movie(movie_filename, fps, bitrate='8192k'):
    @_plot.yield_screenshots
    def generate_movie_():
        h = _plot.height * 2
        w = int(h * 16/9) & ~0x0f
        with FFmpegFrameWriter(movie_filename, fps=fps, size=(w, h), bitrate=bitrate, stdout=True) as writer:
            global sequence_stop
            sequence_stop = False
            for seq in sequence_movie(fps):
                if sequence_stop:
                    with output_state:
                        print('generating movie stopped.')
                    break
                update_movie_camera_pos(*seq['camera_pos'])
                update_movie_color_range_list(*seq['color_range_list'])
                update_movie_plane(*seq['plane'])
                update_movie_alpha_blending(seq['alpha_blending'])
                _plot.fetch_screenshot()
                screenshot = yield
                img = np.asarray(Image.open(io.BytesIO(screenshot)))
                if img.shape[1] > w:
                    offset = (img.shape[1] - w) // 2
                    writer.add(img[:, offset:(offset+w), :3])
                else:
                    offset = (w - img.shape[1]) // 2
                    writer.frame[:, offset:(offset+img.shape[1]), :] = img[:, :, :3]
                    writer.add_frame()
    _plot.screenshot = ''
    print_state()
    with output_state:
        print('generating movie started.')
    generate_movie_()
