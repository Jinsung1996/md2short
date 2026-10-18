import { AbsoluteFill, Audio, Img, Sequence, staticFile, useVideoConfig } from "remotion";
import { BottomBar } from "./BottomBar";
import { BrandHeader } from "./BrandHeader";
import { Captions } from "./Captions";
import { NumberCards } from "./NumberCard";
import { OpeningBriefing } from "./OpeningBriefing";
import { Outro } from "./Outro";
import { VisualLayer } from "./VisualLayer";
import { HEADER_HEIGHT } from "./layout";
import { msToFrames, normalizeColor, type CueProps } from "./types";

/**
 * Composition root. Renders strictly what props.json says — no timing
 * estimation, no editing decisions (AGENTS.md section 3.1).
 */
export const TechExplainer: React.FC<CueProps> = ({
  audioFileName,
  wordsFileName,
  bgColor,
  chunkSize,
  emoticonCues,
  captionSegments,
  companyLogoCues,
  numberCards,
  highlightKeywords,
  openingText,
  openingDurationMs,
  showOpening,
  ctaText,
  showOutro,
}) => {
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill style={{ backgroundColor: normalizeColor(bgColor) }}>
      {/* public/font/font.css — SeoulAlrim / SeoulNotice face declarations. */}
      <link rel="stylesheet" href={staticFile("font/font.css")} />

      <Audio src={staticFile(audioFileName)} />

      <VisualLayer cues={emoticonCues} />

      {companyLogoCues.map((cue, index) => (
        <Sequence
          key={`logo-${index}-${cue.startMs}`}
          from={msToFrames(cue.startMs, fps)}
          durationInFrames={Math.max(1, msToFrames(cue.durationMs, fps))}
          layout="none"
        >
          <Img
            src={staticFile(`company/${cue.logoFileName}`)}
            style={{
              position: "absolute",
              top: HEADER_HEIGHT + 40,
              right: 48,
              width: 200,
              objectFit: "contain",
            }}
          />
        </Sequence>
      ))}

      <NumberCards cards={numberCards} />

      <BrandHeader suppressBeforeMs={showOpening ? openingDurationMs : 0} />
      <BottomBar />

      <Captions
        wordsFileName={wordsFileName}
        chunkSize={chunkSize}
        captionSegments={captionSegments}
        highlightKeywords={highlightKeywords}
        suppressBeforeMs={showOpening ? openingDurationMs : 0}
      />

      {showOpening ? (
        <OpeningBriefing
          openingText={openingText}
          openingDurationMs={openingDurationMs}
        />
      ) : null}

      {showOutro ? <Outro ctaText={ctaText} /> : null}
    </AbsoluteFill>
  );
};
