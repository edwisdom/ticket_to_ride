let ctx: CanvasRenderingContext2D | null | undefined;

/** The rendered width of `text` in a CSS `font`, in the font's own units (px = board units). */
export function textWidth(text: string, font: string): number {
  ctx ??= document.createElement('canvas').getContext('2d');
  if (!ctx) return text.length * 0.6 * (parseFloat(font.split(' ')[1] ?? '') || 13);
  ctx.font = font;
  return ctx.measureText(text).width;
}
