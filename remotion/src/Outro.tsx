import {
  AbsoluteFill,
  Sequence,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { FONT_FAMILY, OUTRO_DURATION_MS } from "./layout";
import { msToFrames } from "./types";

const OutroOverlay: React.FC<{ ctaText: string }> = ({ ctaText }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const pop = spring({ frame, fps, config: { damping: 13, mass: 0.6 } });
  const scale = interpolate(pop, [0, 1], [0.7, 1]);
  const opacity = interpolate(frame, [0, 5], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#000000",
        alignItems: "center",
        justifyContent: "center",
        opacity,
      }}
    >
      <div
        style={{
          transform: `scale(${scale})`,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 40,
        }}
      >
        <span style={{ fontSize: 220 }}>🐯</span>
        <span
          style={{
            fontFamily: FONT_FAMILY,
            fontSize: 132,
            fontWeight: 900,
            letterSpacing: -4,
            color: "#FFD400",
            textAlign: "center",
          }}
        >
          {ctaText}
        </span>
      </div>
    </AbsoluteFill>
  );
};

/**
 * Full-screen subscribe CTA for the last 2.6s (architecture.md section 5.10).
 * Only mounted by TechExplainer when `showOutro` is true (default false).
 */
export const Outro: React.FC<{ ctaText: string }> = ({ ctaText }) => {
  const { fps, durationInFrames } = useVideoConfig();
  const outroFrames = msToFrames(OUTRO_DURATION_MS, fps);
  const from = Math.max(0, durationInFrames - outroFrames);

  return (
    <Sequence
      from={from}
      durationInFrames={durationInFrames - from}
      layout="none"
    >
      <OutroOverlay ctaText={ctaText} />
    </Sequence>
  );
};
