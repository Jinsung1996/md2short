import { BOTTOM_BAR_HEIGHT } from "./layout";

/**
 * Bottom 460px black caption bar (docs/architecture.md section 2.3).
 * Captions.tsx draws the text inside this band.
 */
export const BottomBar: React.FC = () => {
  return (
    <div
      style={{
        position: "absolute",
        bottom: 0,
        left: 0,
        width: "100%",
        height: BOTTOM_BAR_HEIGHT,
        backgroundColor: "#000000",
      }}
    />
  );
};
