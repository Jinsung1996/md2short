import {
  Img,
  Sequence,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { SAFE_HEIGHT, SAFE_TOP, SAFE_WIDTH } from "./layout";
import { msToFrames, type EmoticonCue } from "./types";

/** Gentle Ken Burns range (architecture.md section 5.8). */
const ZOOM_MIN = 1;
const ZOOM_MAX = 1.08;
/** Square side used for `contain` emoticon rendering inside the safe area. */
const EMOTICON_SIZE = Math.min(SAFE_WIDTH, SAFE_HEIGHT) - 90;

/**
 * `emoticonFileName` is a path relative to public/ when it contains a slash
 * (photo mode stages `jobs/<id>/photos/...`), otherwise a bare emoticon PNG
 * name inside public/emoticon/ (architecture.md section 1.7).
 */
const resolveSrc = (fileName: string): string =>
  staticFile(fileName.includes("/") ? fileName : `emoticon/${fileName}`);

const VisualCue: React.FC<{
  cue: EmoticonCue;
  durationInFrames: number;
  zoomIn: boolean;
}> = ({ cue, durationInFrames, zoomIn }) => {
  const frame = useCurrentFrame();
  const progress = interpolate(frame, [0, Math.max(1, durationInFrames - 1)], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const scale = zoomIn
    ? interpolate(progress, [0, 1], [ZOOM_MIN, ZOOM_MAX])
    : interpolate(progress, [0, 1], [ZOOM_MAX, ZOOM_MIN]);
  const isPhoto = cue.assetKind === "photo";

  return (
    <div
      style={{
        position: "absolute",
        top: SAFE_TOP,
        left: 0,
        width: SAFE_WIDTH,
        height: SAFE_HEIGHT,
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Img
        src={resolveSrc(cue.emoticonFileName)}
        style={
          isPhoto
            ? {
                // photo mode: fill the safe content rectangle, crop overflow.
                width: "100%",
                height: "100%",
                objectFit: "cover",
                transform: `scale(${scale})`,
              }
            : {
                // emoticon mode: square, fully contained, never cropped.
                width: EMOTICON_SIZE,
                height: EMOTICON_SIZE,
                objectFit: "contain",
                transform: `scale(${scale})`,
              }
        }
      />
    </div>
  );
};

export const VisualLayer: React.FC<{ cues: EmoticonCue[] }> = ({ cues }) => {
  const { fps } = useVideoConfig();

  return (
    <>
      {cues.map((cue, index) => {
        const durationInFrames = Math.max(1, msToFrames(cue.durationMs, fps));
        return (
          <Sequence
            key={`cue-${index}-${cue.startMs}`}
            from={msToFrames(cue.startMs, fps)}
            durationInFrames={durationInFrames}
            layout="none"
          >
            <VisualCue
              cue={cue}
              durationInFrames={durationInFrames}
              zoomIn={index % 2 === 0}
            />
          </Sequence>
        );
      })}
    </>
  );
};
