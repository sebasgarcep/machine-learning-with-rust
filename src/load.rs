use std::fs::File;
use std::io::Read;
use shared::DataPoint;
use rulinalg::vector::Vector;

fn from_4u8_to_u32 (arr: &[u8]) -> u32 {
    ((arr[0] as u32) << 24) |
    ((arr[1] as u32) << 16) |
    ((arr[2] as u32) << 8) |
    ((arr[3] as u32) << 0)
}

macro_rules! take_4_bytes {
    ($T:ty, $bytes:ident) => (
        from_4u8_to_u32 (&[
            $bytes.next().unwrap().unwrap(),
            $bytes.next().unwrap().unwrap(),
            $bytes.next().unwrap().unwrap(),
            $bytes.next().unwrap().unwrap(),
        ]) as $T
    )
}

macro_rules! take_byte {
    ($bytes:ident) => (
        $bytes.next().unwrap().unwrap()
    )
}

pub fn load_mnist_data () -> Vec<DataPoint> {
    let mut label_bytes = File::open("./mnist/train-labels.idx1-ubyte").unwrap().bytes();
    let magic_number_labels = take_4_bytes!(u32, label_bytes);
    let dataset_size = take_4_bytes!(usize, label_bytes);

    let mut image_file = File::open("./mnist/train-images.idx3-ubyte").unwrap();
    let file_size = image_file.metadata().unwrap().len() as usize;
    let mut image_buffer = Vec::with_capacity(file_size);
    let _ = image_file.read_to_end(&mut image_buffer);

    let magic_number_images = from_4u8_to_u32(&image_buffer[0..4]);
    let dataset_size2 = from_4u8_to_u32(&image_buffer[4..8]) as usize;
    let images_rows = from_4u8_to_u32(&image_buffer[8..12]) as usize;
    let images_cols = from_4u8_to_u32(&image_buffer[12..16]) as usize;
    let len = images_rows * images_cols;

    // assertions
    assert_eq!(magic_number_labels, 0x00000801);
    assert_eq!(magic_number_images, 0x00000803);
    assert_eq!(dataset_size, dataset_size2);
    assert_eq!(images_rows, images_cols);
    assert_eq!(images_rows, 28);

    let mut dataset = Vec::with_capacity(dataset_size);

    for num in 0..dataset_size {
        let label = take_byte!(label_bytes) as usize;
        let mut label_vector = Vector::zeros(10);
        label_vector[label] = 1.0;
        let index = 16 + num * len;
        let digit = image_buffer[index..index+len].iter().map(|&b| (b as f64) / 255.0).collect();
        dataset.push((digit, label_vector));
    }

    assert_eq!(label_bytes.count(), 0);

    dataset
}
