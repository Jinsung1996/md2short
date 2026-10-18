/** Fixed 9:16 layout constants (docs/architecture.md section 2.3). */
export const VIDEO_WIDTH = 1080;
export const VIDEO_HEIGHT = 1920;
export const FPS = 30;

/** Top black bar: 250px safe margin + 220px logo area. */
export const HEADER_SAFE_MARGIN = 250;
export const HEADER_LOGO_HEIGHT = 220;
export const HEADER_HEIGHT = HEADER_SAFE_MARGIN + HEADER_LOGO_HEIGHT; // 470

/** Bottom black caption bar. */
export const BOTTOM_BAR_HEIGHT = 460;

/** Safe content rectangle between the two black bars. */
export const SAFE_TOP = HEADER_HEIGHT;
export const SAFE_HEIGHT = VIDEO_HEIGHT - HEADER_HEIGHT - BOTTOM_BAR_HEIGHT; // 990
export const SAFE_WIDTH = VIDEO_WIDTH;

/** Font family stack provided by public/font/font.css. */
export const FONT_FAMILY =
  '"SeoulAlrim", "SeoulNotice", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif';

/** Outro overlay length (architecture.md section 5.10). */
export const OUTRO_DURATION_MS = 2600;
