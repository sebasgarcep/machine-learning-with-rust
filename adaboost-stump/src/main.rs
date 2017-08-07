// FIND v4l2 COMPATIBLE SPECS: v4l2-ctl --list-formats-ext
// Thanks to https://github.com/oli-obk/camera_capture for most of the code

extern crate piston_window;
extern crate image;
extern crate rscam;

use std::thread::sleep;
use std::time::Duration;
use piston_window::{PistonWindow, Texture, WindowSettings, TextureSettings, clear};
use image::load_from_memory;
use image::{ConvertBuffer, DynamicImage};

fn main() {
    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;

    let mut window: PistonWindow = WindowSettings::new("piston: image", [WIDTH, HEIGHT])
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut tex: Option<Texture<_>> = None;
    let (sender, receiver) = std::sync::mpsc::channel();
    let imgthread = std::thread::spawn(move || {
        let mut camera = rscam::new("/dev/video0").unwrap();
        camera.start(&rscam::Config {
                interval: (1, 30), // 30 fps.
                format: b"MJPG",
                resolution: (WIDTH, HEIGHT),
                ..Default::default()
            })
            .unwrap();

        loop {
            let frame = camera.capture().unwrap();
            let img = match load_from_memory(&frame).unwrap() {
                    DynamicImage::ImageRgb8(rgb_image) => rgb_image,
                    _ => panic!(),
                }
                .convert();

            if let Err(_) = sender.send(img) {
                break;
            }

            sleep(Duration::from_millis(50));
        }
    });

    while let Some(e) = window.next() {
        if let Ok(img) = receiver.try_recv() {
            if let Some(mut t) = tex {
                t.update(&mut window.encoder, &img).unwrap();
                tex = Some(t);
            } else {
                tex = Texture::from_image(&mut window.factory, &img, &TextureSettings::new()).ok();
            }
        }
        window.draw_2d(&e, |c, g| {
            clear([1.0; 4], g);
            if let Some(ref t) = tex {
                piston_window::image(t, c.transform, g);
            }
        });
    }

    drop(receiver);
    imgthread.join().unwrap();
}
