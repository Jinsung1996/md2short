/**
 * Shared data contract between the Python layer (md2short.render -> props.json)
 * and the Remotion composition. Mirrors docs/architecture.md section 4.2 / 4.3.
 */

/** One Scribe word timestamp. `start`/`end` are milliseconds. */
export type Word = {
  text: string;
  start: number;
  end: number;
};

/** transcripts/words.json — the single source of truth for all timing. */
export type WordsDoc = {
  language_code?: string;
  audio_duration_ms?: number;
  text?: string;
  words: Word[];
};

/** `emoticon` = deterministic tiger PNG, `photo` = generated 9:16 photo. */
export type AssetKind = "emoticon" | "photo";
export type VisualMode = "emoticon" | "photo";

/**
 * One visual cue. The `emoticonFileName` field name is kept for both asset
 * kinds (architecture.md section 1.7) — it is a path relative to `public/`
 * when it contains a slash (photo mode: `jobs/<id>/photos/seg_NN_<sha1>.png`)
 * and a bare file name inside `public/emoticon/` otherwise.
 */
export type EmoticonCue = {
  emoticonFileName: string;
  startMs: number;
  durationMs: number;
  assetKind: AssetKind;
  visualMode: VisualMode;
  keyword?: string | null;
  /** photo mode only — extracted subject the image prompt was focused on. */
  primaryVisualSubject?: string | null;
  /** photo mode only — supporting visual keywords for the image prompt. */
  visualKeywords?: string[];
};

/**
 * Caption chunk boundary shared with the visual cues. When present, Captions
 * locks onto these boundaries instead of recomputing them from words.json.
 */
export type CaptionSegment = {
  startMs: number;
  endMs: number;
  text?: string;
};

/** Company logo overlay cue (`public/company/<fileName>`). */
export type CompanyLogoCue = {
  logoFileName: string;
  startMs: number;
  durationMs: number;
  company?: string;
};

/** Number sticker card cue (architecture.md section 5.9). */
export type NumberCard = {
  text: string;
  startMs: number;
  durationMs: number;
  label?: string | null;
};

/** Full composition props — matches props.json in architecture.md section 4.3. */
export type CueProps = {
  audioFileName: string;
  wordsFileName: string;
  bgColor: string;
  chunkSize: number;
  emoticonCues: EmoticonCue[];
  captionSegments: CaptionSegment[];
  companyLogoCues: CompanyLogoCue[];
  numberCards: NumberCard[];
  highlightKeywords: string[];
  openingText: string;
  openingDurationMs: number;
  showOpening: boolean;
  ctaText: string;
  showOutro: boolean;
  durationInFrames: number;
};

/** Single ms -> frame helper. AGENTS.md section 4.2 forbids ad-hoc variants. */
export const msToFrames = (ms: number, fps: number): number =>
  Math.round((ms / 1000) * fps);

/** `0xFFFFFF` (JobConfig) and `#FFFFFF` (props.json) both occur. */
export const normalizeColor = (color: string): string =>
  color.startsWith("0x") ? `#${color.slice(2)}` : color;
