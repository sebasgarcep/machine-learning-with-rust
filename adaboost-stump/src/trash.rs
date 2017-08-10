use std::collections::HashMap;

#[allow(dead_code)]
fn get_faces_map() -> HashMap<String, Vec<FaceFeature>> {
    let mut faces_map = HashMap::new();
    let faces_file = File::open("./data/faces.txt").unwrap();
    let faces_buffer = BufReader::new(&faces_file);
    for line in faces_buffer.lines() {
        let l = line.unwrap();
        let tokens: Vec<&str> = l.split(" ").collect();
        let filename = tokens[0];

        let coords: Vec<i64> = tokens[1..]
            .iter()
            .map(|s| {
                let parts: Vec<&str> = s.split(".").collect();
                let integer_part = parts[0];
                integer_part.parse().unwrap()
            })
            .collect();

        let feature = FaceFeature {
            left_eye: (coords[0], coords[1]),
            right_eye: (coords[2], coords[3]),
            nose: (coords[4], coords[5]),
            left_mouth: (coords[6], coords[7]),
            center_mouth: (coords[8], coords[9]),
            right_mouth: (coords[10], coords[11]),
        };

        let keyname = filename.to_string();
        let feature_list = faces_map.entry(keyname).or_insert(Vec::new());

        feature_list.push(feature);
    }

    faces_map
}

#[derive(Debug)]
struct FaceFeature {
    left_eye: (i64, i64),
    right_eye: (i64, i64),
    nose: (i64, i64),
    left_mouth: (i64, i64),
    center_mouth: (i64, i64),
    right_mouth: (i64, i64),
}
