import atexit
import copy
import logging
import msgpack
import threading
import time
from base64 import b64decode
from flask import Flask, send_from_directory
from werkzeug import Response
from werkzeug.serving import make_server
from playwright.sync_api import sync_playwright
import imageio
import time as timing
from k3d.helpers import to_json
import k3d
import os
import numpy as np
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# see https://peter.sh/experiments/chromium-command-line-switches/#enable-gpu
chrome_args = ["--enable-gpu","--enable-gpu-rasterization", "-enable-unsafe-webgpu" , "--use-gl=egl",
               "--enable-zero-copy", "--ignore-gpu-blocklist", "--enable-features=Vulkan"]

class k3d_remote:
    def __init__(self, k3d_plot, driver, width=1280, height=720, port=0):

        #driver.set_window_size(width, height)

        self.port = port
        self.browser = driver
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.k3d_plot = k3d_plot

        self.api = Flask(__name__)

        self.server = make_server("localhost", port=port, app=self.api)
        port = self.server.port
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(), daemon=True)
        self.thread.deamon = True
        self.thread.start()

        self.synced_plot = {k: None for k in k3d_plot.get_plot_params().keys()}
        self.synced_objects = {}

        @self.api.route('/<path:path>')
        def static_file(path):
            root_dir = self.k3d_plot.get_static_path()
            return send_from_directory(root_dir, path)

        @self.api.route('/ping')
        def ping():
            return Response(":)")

        @self.api.route('/', methods=['POST'])
        def generate():
            current_plot_params = self.k3d_plot.get_plot_params()
            plot_diff = {k: current_plot_params[k] for k in current_plot_params.keys()
                         if current_plot_params[k] != self.synced_plot[k] and k != 'minimumFps'}

            objects_diff = {}
            def get_t(data):
                if isinstance(data, dict):
                    return data[str(self.k3d_plot.time)]
                return data

            for o in self.k3d_plot.objects:
                if o.id not in self.synced_objects:
                    objects_diff[o.id] = {k: to_json(k, get_t(o[k]), o) for k in o.keys if
                                          not k.startswith('_')}
                else:
                    for p in o.keys:
                        if p.startswith('_'):
                            continue

                        if p == 'voxels_group':
                            sync = True  # todo
                        else:
                            try:
                                sync = (get_t(o[p]) != get_t(self.synced_objects[o.id][p])).any()
                            except Exception:
                                sync = o[p] != self.synced_objects[o.id][p]

                        if sync:
                            if o.id not in objects_diff.keys():
                                objects_diff[o.id] = {"id": o.id, "type": o.type}

                            objects_diff[o.id][p] = to_json(p, get_t(o[p]), o)

            for k in self.synced_objects.keys():
                if k not in self.k3d_plot.object_ids:
                    objects_diff[k] = None  # to remove from plot

            diff = {
                "plot_diff": plot_diff,
                "objects_diff": objects_diff
            }

            self.synced_objects = {v.id: {k: copy.deepcopy(get_t(v[k])) for k in v.keys} for v in
                                   self.k3d_plot.objects}
            self.synced_plot = current_plot_params

            return Response(msgpack.packb(diff, use_bin_type=True),
                            mimetype='application/octet-stream')

        while self.page.evaluate("typeof(window.headlessK3D) !== 'undefined'") == False:
            time.sleep(0.1)
            self.page.goto("http://localhost:" + str(port) + "/headless.html")

        atexit.register(self.close)

    def sync(self, hold_until_refreshed=True):
        self.page.evaluate("k3dRefresh()")
        if hold_until_refreshed:
            while self.page.evaluate("window.refreshed") == False:
                time.sleep(0.1)

    def get_browser_screenshot(self):
        return self.page.screenshot()

    def camera_reset(self, factor=1.5):
        self.page.evaluate("K3DInstance.resetCamera(%f)" % factor)
        # refresh dom elements
        self.page.evaluate("K3DInstance.refreshGrid()")
        self.page.evaluate("K3DInstance.dispatch(K3DInstance.events.RENDERED)")

    def get_screenshot(self, only_canvas=False):
        screenshot = self.page.evaluate("""
        K3DInstance.getScreenshot(K3DInstance.parameters.screenshotScale, %d).then(function (d){
        return d.toDataURL().split(',')[1];
        });                                 
        """ % only_canvas)

        return b64decode(screenshot)

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server = None
        #self.context.close()

def test_aquarium():
    from playwright.sync_api import sync_playwright
    import time
    with sync_playwright() as p:
        browser = p.chromium.launch(args=chrome_args)
        context = browser.new_context(record_video_dir="videos/")
        page = context.new_page()
        page.goto("https://webglsamples.github.io/aquarium/aquarium.html?numFish=50000")
        time.sleep(1)
        page.screenshot(path="aquarium.png")
        context.close()

def test_hardware_acceleration(screenshot=True):
    print(chrome_args)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=chrome_args)
        page = browser.new_page()
        page.goto("chrome://gpu")
        #page.goto("https://webglreport.com/?v=1")
        loc = page.locator(".feature-status-list")
        text = None
        #text = loc.all_inner_texts()[0].split("\n")
        #for t in text:
        #    if "WebGL:" in t:
        #        print(t)
        if screenshot:
            #time.sleep(1)
            page.screenshot(path="gpu_settings.png")
        print(chrome_args)
        return

def generate_screenshots(objects, filename, fov=None, times=None, html_slow_down=1,
                         background_color=0x000000, camera=None):
    pl = k3d.plot(
        camera_rotate_speed=3,
        camera_zoom_speed=5,
        screenshot_scale=2,
        background_color=background_color,
        grid_visible=False,
        camera_auto_fit=True,
        axes_helper=False,
        lighting=2
        )
    for o in objects:
        pl += o
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename + ".html", 'w') as f:
        f.write(pl.get_snapshot())
    with sync_playwright() as p:
        browser = p.chromium.launch(args = chrome_args)
        headless = k3d_remote(pl, browser, port=0)
        headless.sync()
        if fov is not None:
            pl.camera_fov = fov
        if camera is not None and len(camera.shape)==1:
            pl.camera = camera
        print(pl.camera)
        headless.sync()
        if times is not None:
            frames= []
            for i,t in enumerate(times):
                if camera is not None and len(camera.shape)==2:
                    pl.camera = camera[i]
                pl.time = t * html_slow_down
                headless.sync(hold_until_refreshed=True)
                img = headless.get_screenshot()
                with open(filename + f"_{t:.3f}.png", 'wb') as f:
                    f.write(img)
        else:
            with open(filename + ".png", 'wb') as f:
                f.write(headless.get_screenshot())
            frames = None 
        headless.close()
        return [] #frames

def write_animation(img_dir, filename, total_duration):
    from PIL import Image
    img = []
    for f in os.listdir(img_dir):
        if f.endswith(".png"):
            img.append(Image.open(f"{img_dir}/{f}"))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img[0].save(f"{filename}", save_all=True, append_images=img[1:],
                duration=int(total_duration / len(img)), loop=0, quality=95,
                optimize=True, minimize_size=True, allow_mixed=True)

