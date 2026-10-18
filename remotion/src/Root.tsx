import { Composition } from "remotion";
import { FPS, VIDEO_HEIGHT, VIDEO_WIDTH } from "./layout";
import { TechExplainer } from "./TechExplainer";
import type { CueProps } from "./types";

const defaultProps: CueProps = {
  audioFileName: "jobs/sample/narration.mp3",
  wordsFileName: "jobs/sample/words.json",
  bgColor: "#FFFFFF",
  chunkSize: 14,
  emoticonCues: [],
  captionSegments: [],
  companyLogoCues: [],
  numberCards: [],
  highlightKeywords: [],
  openingText: "",
  openingDurationMs: 0,
  showOpening: false,
  ctaText: "구독 + 알람",
  showOutro: false,
  durationInFrames: 900,
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="TechExplainer"
      component={TechExplainer}
      width={VIDEO_WIDTH}
      height={VIDEO_HEIGHT}
      fps={FPS}
      durationInFrames={defaultProps.durationInFrames}
      defaultProps={defaultProps}
      // Duration always comes from the real TTS audio length, which the Python
      // layer writes into props.json (architecture.md section 1.4).
      calculateMetadata={({ props }) => ({
        durationInFrames: Math.max(
          1,
          Math.round(props.durationInFrames ?? defaultProps.durationInFrames),
        ),
      })}
    />
  );
};
