import {
  Sequence,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { FONT_FAMILY, SAFE_TOP } from "./layout";
import { msToFrames, type NumberCard as NumberCardCue } from "./types";

/** Default card lifetime (architecture.md section 5.9). */
const DEFAULT_DURATION_MS = 1800;
const TILT_DEG = -4;

export const NumberCard: React.FC<{ card: NumberCardCue }> = ({ card }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const pop = spring({ frame, fps, config: { damping: 12, mass: 0.6 } });
  const scale = interpolate(pop, [0, 1], [0.6, 1]);
  const opacity = interpolate(pop, [0, 0.35], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        top: SAFE_TOP + 70,
        right: 70,
        transform: `scale(${scale}) rotate(${TILT_DEG}deg)`,
        transformOrigin: "top right",
        opacity,
        backgroundColor: "#FFD400",
        color: "#111111",
        borderRadius: 32,
        border: "8px solid #111111",
        boxShadow: "14px 14px 0 rgba(17,17,17,0.35)",
        padding: "28px 44px",
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: 6,
        maxWidth: 620,
      }}
    >
      {card.label ? (
        <span
          style={{
            fontFamily: FONT_FAMILY,
            fontSize: 44,
            fontWeight: 700,
            opacity: 0.8,
          }}
        >
          {card.label}
        </span>
      ) : null}
      <span
        style={{
          fontFamily: FONT_FAMILY,
          fontSize: 104,
          fontWeight: 900,
          lineHeight: 1.05,
          letterSpacing: -2,
        }}
      >
        {card.text}
      </span>
    </div>
  );
};

export const NumberCards: React.FC<{ cards: NumberCardCue[] }> = ({ cards }) => {
  const { fps } = useVideoConfig();

  return (
    <>
      {cards.map((card, index) => (
        <Sequence
          key={`number-${index}-${card.startMs}`}
          from={msToFrames(card.startMs, fps)}
          durationInFrames={Math.max(
            1,
            msToFrames(card.durationMs || DEFAULT_DURATION_MS, fps),
          )}
          layout="none"
        >
          <NumberCard card={card} />
        </Sequence>
      ))}
    </>
  );
};
