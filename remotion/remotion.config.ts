import { Config } from "@remotion/cli/config";

// MD2Short renders 9:16 shortform video. The pipeline (md2short.render) passes
// --codec/--width/--height explicitly, these are the local-preview defaults.
Config.setVideoImageFormat("jpeg");
Config.setCodec("h264");
Config.setOverwriteOutput(true);
