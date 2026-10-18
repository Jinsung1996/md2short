import { useState } from "react";
import { staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import {
  FONT_FAMILY,
  HEADER_HEIGHT,
  HEADER_LOGO_HEIGHT,
  HEADER_SAFE_MARGIN,
} from "./layout";
import { msToFrames } from "./types";

/**
 * Top 470px black bar = 250px safe margin + 220px logo area
 * (docs/architecture.md section 2.3).
 *
 * A plain <img> is used instead of Remotion's <Img> on purpose: the logo asset
 * is staged by the Python layer and may be absent during local preview, and a
 * plain tag degrades to the text wordmark instead of failing the render.
 */
export const BrandHeader: React.FC<{
  /** Opening briefing interval — suppressed so its wordmark fallback doesn't duplicate the opening box's text (section 5.10). */
  suppressBeforeMs: number;
}> = ({ suppressBeforeMs }) => {
  const [logoFailed, setLogoFailed] = useState(false);
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  if (frame < msToFrames(suppressBeforeMs, fps)) {
    return null;
  }

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: HEADER_HEIGHT,
        backgroundColor: "#000000",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: HEADER_SAFE_MARGIN,
          left: 0,
          width: "100%",
          height: HEADER_LOGO_HEIGHT,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {logoFailed ? (
          <span
            style={{
              fontFamily: FONT_FAMILY,
              fontSize: 96,
              fontWeight: 800,
              color: "#FFFFFF",
              letterSpacing: -2,
            }}
          >
            데이브리핑
          </span>
        ) : (
          <img
            src={staticFile("img/header_logo.png")}
            onError={() => setLogoFailed(true)}
            style={{
              maxHeight: HEADER_LOGO_HEIGHT - 40,
              maxWidth: 900,
              objectFit: "contain",
            }}
          />
        )}
      </div>
    </div>
  );
};
