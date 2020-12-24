# paddle-sys-demo

Demos for paddle-sys. 

## Demo

### ResNet

This demo showed how to run the residual network model in Rust with paddle-sys. Please replace the paths of the model and parameters in the source code. And add your pre-compiled library to the environment variable before running.

```bash
cargo build
export LD_LIBRARY_PATH=/path/to/paddle_lib:$LD_LIBRARY_PATH
cargo run
```

## License

paddle-sys-demo is provided under the <a href="LICENSE">Apache License, Version 2.0</a>.