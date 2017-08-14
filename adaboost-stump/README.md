# AdaBoost-Decision Stump Viola Jones (2001)

### To train the model:
```bash
cargo run --bin train --release
```

### To see static results:
```bash
cargo run --bin validate --release
```

### TODO:
- Real Time Face Detection
  - Doing detection on every frame
  - Doing some magic to scale sizes to ones the predictor understands (19x19).
