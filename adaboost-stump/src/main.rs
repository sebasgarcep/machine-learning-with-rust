/**
*
*   Dependencies:
*
*   gdk-pixbuf = "0.1.3"
*   glib = "0.1.3"
*   gtk = "0.1.3"
*   rscam = "0.5.3"
*
*/

extern crate rscam;
extern crate gtk;
extern crate gdk_pixbuf;
extern crate glib;

use std::fs;
use std::io::Write;
use std::thread::sleep;
use std::cell::RefCell;
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;
use std::thread;
use rscam::{Camera, Config};
use gtk::prelude::*;
use gtk::{Button, Image, Window, WindowType};
use gdk_pixbuf::Pixbuf;
use glib::{Continue, timeout_add};

fn get_pixbuf_from_camera(camera: &Camera) -> Pixbuf {
    let frame = camera.capture().unwrap();
    let (pic_width, pic_height) = frame.resolution;

    let colorspace = 0;
    let rowstride = 3 * pic_width;
    let vec = Vec::from(&frame[..]);

    Pixbuf::new_from_vec(vec,
                         colorspace,
                         false,
                         8,
                         pic_width as i32,
                         pic_height as i32,
                         rowstride as i32)
}

fn receive() -> glib::Continue {
    GLOBAL.with(|global| {
        if let Some((ref camera, ref img)) = *global.borrow() {
            let pixbuf = get_pixbuf_from_camera(&camera);
            img.set_from_pixbuf(Some(&pixbuf));
        }
    });
    glib::Continue(false)
}

fn main() {
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 600;

    if gtk::init().is_err() {
        println!("Failed to initialize GTK.");
        return;
    }

    let window = Window::new(WindowType::Toplevel);
    window.set_title("First GTK+ Program");
    window.set_default_size(WIDTH as i32, HEIGHT as i32);

    let mut camera = rscam::new("/dev/video0").unwrap();
    camera.start(&rscam::Config {
            interval: (1, 30), // 30 fps.
            resolution: (1280, 720),
            format: b"YUYV",
            ..Default::default()
        })
        .unwrap();

    let pixbuf = get_pixbuf_from_camera(&camera);
    let img = Image::new_from_pixbuf(Some(&pixbuf));
    window.add(&img);

    // put TextBuffer and receiver in thread local storage
    GLOBAL.with(move |global| *global.borrow_mut() = Some((camera, img)));

    thread::spawn(move || {
        loop {
            // do long work
            thread::sleep(Duration::from_millis(50));
            // receive will be run on the main thread
            glib::idle_add(receive);
        }
    });

    window.show_all();

    window.connect_delete_event(|_, _| {
        gtk::main_quit();
        Inhibit(false)
    });

    gtk::main();
}

// declare a new thread local storage key
thread_local!(
    static GLOBAL: RefCell<Option<(Camera, Image)>> = RefCell::new(None)
);
