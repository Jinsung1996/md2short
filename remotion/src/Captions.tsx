import { useEffect, useMemo, useState } from "react";
import {
  cancelRender,
  continueRender,
  delayRender,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { BOTTOM_BAR_HEIGHT, FONT_FAMILY, VIDEO_WIDTH } from "./layout";
import type { CaptionSegment, Word, WordsDoc } from "./types";

/** Sentence-end punctuation that closes a caption chunk (section 5.4). */
const SENTENCE_END = /[.!?。！？…]+["'”’)\]]*$/;
/** Inter-word silence that closes a caption chunk (section 5.4). */
const GAP_BOUNDARY_MS = 600;
/** chunkSize is a safety cap on words per chunk, not the primary rule. */
const CHUNK_SIZE_MIN = 1;
const CHUNK_SIZE_MAX = 24;
const MAX_LINES = 2;

const CAPTION_PADDING_X = 56;
const CAPTION_MAX_WIDTH = VIDEO_WIDTH - CAPTION_PADDING_X * 2;
/** Tried largest-first; the first size that fits 2 lines wins (section 5.4). */
const FONT_SIZES = [78, 72, 66, 60, 54, 48, 44];
const WORD_GAP_UNITS = 0.45;

type Chunk = {
  words: Word[];
  startMs: number;
  endMs: number;
};

/** Rough advance width in em units: CJK/Hangul is full width, latin is not. */
const charUnits = (char: string): number =>
  /[ᄀ-ᇿ⺀-鿿가-힯＀-￯]/.test(char)
    ? 1
    : 0.55;

const textUnits = (text: string): number =>
  Array.from(text).reduce((sum, char) => sum + charUnits(char), 0);

/**
 * True if a *real* sentence/gap boundary lands within `maxExtra` more words,
 * scanning forward from `startIndex` (not yet consumed). Mirrors Python
 * `caption_plan._sentence_ends_within()` -- running off the end of `words`
 * without finding one is NOT a boundary: `buildChunksFromWords` only ever
 * calls this while a further word is already known to exist (`!next` always
 * fires first and independently), so treating "ran out of lookahead budget"
 * as "found a boundary" would defer cuts far past what `cap` is meant to
 * allow for a punctuation-free run of words.
 */
const sentenceEndsWithin = (words: Word[], startIndex: number, maxExtra: number): boolean => {
  for (let offset = 0; offset < maxExtra; offset += 1) {
    const index = startIndex + offset;
    if (index >= words.length) {
      return false;
    }
    const word = words[index];
    if (SENTENCE_END.test(word.text.trim())) {
      return true;
    }
    const next = words[index + 1];
    if (next && next.start - word.end >= GAP_BOUNDARY_MS) {
      return true;
    }
  }
  return false;
};

/**
 * `cap` is a safety upper bound for an abnormally long run-on sentence, not
 * the primary rule. A plain word-count cut right at `cap` can land a few
 * words before an ordinary sentence would have ended anyway, leaving a
 * short, visually jarring tail chunk. So once the cap is hit, a cut is
 * deferred (up to `CHUNK_SIZE_MAX` words total) when the sentence is about
 * to end anyway; a genuinely long run-on still gets cut, just never later
 * than `CHUNK_SIZE_MAX`. Mirrors Python `caption_plan.split_word_spans()`.
 */
const buildChunksFromWords = (words: Word[], cap: number): Word[][] => {
  const chunks: Word[][] = [];
  let current: Word[] = [];

  words.forEach((word, index) => {
    current.push(word);
    const next = words[index + 1];
    const endsSentence = SENTENCE_END.test(word.text.trim());
    const gapBoundary = next ? next.start - word.end >= GAP_BOUNDARY_MS : true;

    if (!next || endsSentence || gapBoundary) {
      chunks.push(current);
      current = [];
      return;
    }

    if (current.length >= cap) {
      const remainingRoom = CHUNK_SIZE_MAX - current.length;
      if (remainingRoom > 0 && sentenceEndsWithin(words, index + 1, remainingRoom)) {
        return;
      }
      chunks.push(current);
      current = [];
    }
  });

  if (current.length > 0) {
    chunks.push(current);
  }
  return chunks;
};

/** captionSegments lock the chunk boundaries to the visual cue timeline. */
const buildChunksFromSegments = (
  words: Word[],
  segments: CaptionSegment[],
): Chunk[] =>
  segments
    .map((segment) => ({
      words: words.filter(
        (word) => word.end > segment.startMs && word.start < segment.endMs,
      ),
      startMs: segment.startMs,
      endMs: segment.endMs,
    }))
    .filter((chunk) => chunk.words.length > 0);

/** Greedy wrap into lines of word indices. */
const wrapIntoLines = (words: Word[], maxUnits: number): number[][] => {
  const lines: number[][] = [];
  let line: number[] = [];
  let used = 0;

  words.forEach((word, index) => {
    const width = textUnits(word.text);
    const needed = line.length === 0 ? width : width + WORD_GAP_UNITS;
    if (line.length > 0 && used + needed > maxUnits) {
      lines.push(line);
      line = [index];
      used = width;
      return;
    }
    line.push(index);
    used += needed;
  });

  if (line.length > 0) {
    lines.push(line);
  }
  return lines;
};

/** Shrink the font until the chunk fits MAX_LINES, else use the smallest. */
const fitChunk = (words: Word[]): { fontSize: number; lines: number[][] } => {
  for (const fontSize of FONT_SIZES) {
    const lines = wrapIntoLines(words, CAPTION_MAX_WIDTH / fontSize);
    if (lines.length <= MAX_LINES) {
      return { fontSize, lines };
    }
  }
  const fontSize = FONT_SIZES[FONT_SIZES.length - 1];
  return {
    fontSize,
    lines: wrapIntoLines(words, CAPTION_MAX_WIDTH / fontSize).slice(0, MAX_LINES),
  };
};

const stripPunctuation = (text: string): string =>
  text.replace(/[.,!?。、，！？…"'”’()\[\]]/g, "").trim();

/** At most one yellow keyword per chunk (section 5.4). */
const findHighlightIndex = (words: Word[], keywords: string[]): number => {
  const sorted = [...keywords]
    .filter((keyword) => keyword.trim().length > 0)
    .sort((a, b) => b.length - a.length);

  for (const keyword of sorted) {
    const index = words.findIndex((word) =>
      stripPunctuation(word.text).includes(keyword),
    );
    if (index >= 0) {
      return index;
    }
  }
  return -1;
};

export const Captions: React.FC<{
  wordsFileName: string;
  chunkSize: number;
  captionSegments: CaptionSegment[];
  highlightKeywords: string[];
  /** Opening briefing interval — those words are not captioned (section 5.10). */
  suppressBeforeMs: number;
}> = ({
  wordsFileName,
  chunkSize,
  captionSegments,
  highlightKeywords,
  suppressBeforeMs,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const [words, setWords] = useState<Word[]>([]);
  const [handle] = useState(() => delayRender(`words.json: ${wordsFileName}`));

  useEffect(() => {
    let cancelled = false;
    fetch(staticFile(wordsFileName))
      .then((response) => {
        if (!response.ok) {
          throw new Error(`${response.status} ${wordsFileName}`);
        }
        return response.json() as Promise<WordsDoc>;
      })
      .then((doc) => {
        if (!cancelled) {
          setWords(Array.isArray(doc.words) ? doc.words : []);
        }
        continueRender(handle);
      })
      .catch((error) => {
        cancelRender(error);
      });
    return () => {
      cancelled = true;
    };
  }, [handle, wordsFileName]);

  const chunks = useMemo<Chunk[]>(() => {
    const captionable = words.filter((word) => word.end > suppressBeforeMs);
    if (captionable.length === 0) {
      return [];
    }

    if (captionSegments.length > 0) {
      return buildChunksFromSegments(captionable, captionSegments);
    }

    const cap = Math.min(
      CHUNK_SIZE_MAX,
      Math.max(CHUNK_SIZE_MIN, Math.round(chunkSize)),
    );
    const grouped = buildChunksFromWords(captionable, cap);
    return grouped.map((group, index) => {
      const next = grouped[index + 1];
      const last = group[group.length - 1];
      return {
        words: group,
        startMs: group[0].start,
        endMs: next ? next[0].start : last.end + 400,
      };
    });
  }, [words, captionSegments, chunkSize, suppressBeforeMs]);

  const timeMs = (frame / fps) * 1000;
  const chunk = chunks.find(
    (candidate) => timeMs >= candidate.startMs && timeMs < candidate.endMs,
  );

  if (!chunk) {
    return null;
  }

  const { fontSize, lines } = fitChunk(chunk.words);
  const highlightIndex = findHighlightIndex(chunk.words, highlightKeywords);

  return (
    <div
      style={{
        position: "absolute",
        bottom: 0,
        left: 0,
        width: "100%",
        height: BOTTOM_BAR_HEIGHT,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: fontSize * 0.26,
        paddingLeft: CAPTION_PADDING_X,
        paddingRight: CAPTION_PADDING_X,
        boxSizing: "border-box",
      }}
    >
      {lines.map((line, lineIndex) => (
        <div
          key={`line-${lineIndex}`}
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "baseline",
            justifyContent: "center",
            gap: fontSize * WORD_GAP_UNITS,
            whiteSpace: "pre",
          }}
        >
          {line.map((wordIndex) => {
            const word = chunk.words[wordIndex];
            const isActive = timeMs >= word.start && timeMs < word.end;
            const scale = isActive
              ? interpolate(
                  timeMs - word.start,
                  [0, 90, 220],
                  [1, 1.14, 1.02],
                  { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
                )
              : 1;

            return (
              <span
                key={`word-${wordIndex}`}
                style={{
                  fontFamily: FONT_FAMILY,
                  fontSize,
                  fontWeight: 800,
                  lineHeight: 1.18,
                  color: wordIndex === highlightIndex ? "#FFD400" : "#FFFFFF",
                  transform: `scale(${scale})`,
                  transformOrigin: "center bottom",
                  display: "inline-block",
                }}
              >
                {word.text}
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
};
