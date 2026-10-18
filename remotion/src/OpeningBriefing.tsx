import {
  Sequence,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { FONT_FAMILY, HEADER_HEIGHT, SAFE_WIDTH } from "./layout";
import { msToFrames } from "./types";

const FADE_OUT_MS = 350;

/**
 * BrandHeader suppresses itself for the whole opening window (TechExplainer
 * passes `suppressBeforeMs={openingDurationMs}`), so the top `HEADER_HEIGHT`
 * (470px) is genuinely empty black space during the opening -- not shared
 * with the logo/wordmark. Anchoring the opening box there, instead of
 * centering it over the full safe rectangle, keeps it off the emoticon/photo
 * art in `VisualLayer` below `SAFE_TOP`. A little slack past `HEADER_HEIGHT`
 * borrows from the image's own ~45px contain margin at the top of the safe
 * rect, which is harmless overlap.
 */
const OPENING_REGION_HEIGHT = HEADER_HEIGHT + 40;

const OpeningBox: React.FC<{ lines: string[]; durationInFrames: number }> = ({
  lines,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const pop = spring({ frame, fps, config: { damping: 14, mass: 0.7 } });
  const scale = interpolate(pop, [0, 1], [0.82, 1]);
  const fadeOutFrames = msToFrames(FADE_OUT_MS, fps);
  const opacity = interpolate(
    frame,
    [0, 6, Math.max(7, durationInFrames - fadeOutFrames), durationInFrames],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: SAFE_WIDTH,
        height: OPENING_REGION_HEIGHT,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        opacity,
      }}
    >
      <div
        style={{
          transform: `scale(${scale})`,
          backgroundColor: "#111111",
          border: "10px solid #FFD400",
          borderRadius: 48,
          padding: "56px 80px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 18,
        }}
      >
        {lines.map((line, index) => (
          <span
            key={`opening-${index}`}
            style={{
              fontFamily: FONT_FAMILY,
              fontSize: index === 0 ? 108 : 132,
              fontWeight: 900,
              lineHeight: 1.1,
              letterSpacing: -3,
              color: index === 0 ? "#FFFFFF" : "#FFD400",
              textAlign: "center",
            }}
          >
            {line}
          </span>
        ))}
      </div>
    </div>
  );
};

/**
 * Opening briefing box, anchored in the empty (header-suppressed) top region
 * rather than over the emoticon/photo art (architecture.md section 5.10).
 * Only mounted by TechExplainer when `showOpening` is true.
 */
export const OpeningBriefing: React.FC<{
  openingText: string;
  openingDurationMs: number;
}> = ({ openingText, openingDurationMs }) => {
  const { fps } = useVideoConfig();
  const lines = openingText.split("\n").filter((line) => line.trim().length > 0);
  const durationInFrames = msToFrames(openingDurationMs, fps);

  if (lines.length === 0 || durationInFrames < 1) {
    return null;
  }

  return (
    <Sequence from={0} durationInFrames={durationInFrames} layout="none">
      <OpeningBox lines={lines} durationInFrames={durationInFrames} />
    </Sequence>
  );
};
