import type { CardColor, RouteColor } from './layout';

export const CARD_COLORS: CardColor[] = ['purple', 'white', 'blue', 'yellow', 'orange', 'black', 'red', 'green'];

interface Swatch {
  fill: string;
  /** Outlines and shading. */
  dark: string;
  /** Ink that reads on `fill`: the symbol's colour. */
  ink: string;
  label: string;
}

export const SWATCH: Record<RouteColor | 'locomotive', Swatch> = {
  purple: { fill: '#b05aa6', dark: '#6c2f66', ink: '#f6e6f3', label: 'Purple' },
  white: { fill: '#f3efe4', dark: '#8f877a', ink: '#5b544b', label: 'White' },
  blue: { fill: '#2f6db3', dark: '#173d6b', ink: '#e3eefb', label: 'Blue' },
  yellow: { fill: '#f1cd3a', dark: '#8d6f0d', ink: '#5f4a06', label: 'Yellow' },
  orange: { fill: '#e78a2c', dark: '#8a4a0c', ink: '#fff1e0', label: 'Orange' },
  black: { fill: '#2e2b29', dark: '#0d0c0b', ink: '#d9d3c7', label: 'Black' },
  red: { fill: '#c83a30', dark: '#6e1a14', ink: '#fbe5e2', label: 'Red' },
  green: { fill: '#4e9a3c', dark: '#24521a', ink: '#e7f5e1', label: 'Green' },
  gray: { fill: '#b9b3a6', dark: '#6b665d', ink: '#57524a', label: 'Any colour' },
  locomotive: { fill: '#d9c08a', dark: '#5a4325', ink: '#3a2b18', label: 'Locomotive' },
};

const circle = (cx: number, cy: number, r: number) =>
  `M${cx - r} ${cy}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0Z`;

const star = (() => {
  const pts = Array.from({ length: 10 }, (_, i) => {
    const r = i % 2 ? 0.42 : 1;
    const a = (i * Math.PI) / 5 - Math.PI / 2;
    return `${(r * Math.cos(a)).toFixed(3)} ${(r * Math.sin(a) + 0.08).toFixed(3)}`;
  });
  return `M${pts.join('L')}Z`;
})();

/** One shape per colour, so no colour is told apart by hue alone. Unit size, centred on 0,0,
 * drawn with fill-rule evenodd. */
export const SYMBOL: Record<CardColor | 'locomotive', string> = {
  red: 'M0 0.92C-0.25 0.7-1 0.22-1-0.3C-1-0.78-0.55-1-0.27-0.86C-0.12-0.79 0-0.66 0-0.58C0-0.66 0.12-0.79 0.27-0.86C0.55-1 1-0.78 1-0.3C1 0.22 0.25 0.7 0 0.92Z',
  orange: 'M0-0.95L0.98 0.78L-0.98 0.78Z',
  yellow: star,
  green: circle(0, -0.46, 0.4) + circle(-0.45, 0.14, 0.4) + circle(0.45, 0.14, 0.4) + 'M-0.12 0.1L0.12 0.1L0.26 0.98L-0.26 0.98Z',
  blue: 'M0-1C0.32-0.5 0.78-0.05 0.78 0.32A0.78 0.78 0 0 1-0.78 0.32C-0.78-0.05-0.32-0.5 0-1Z',
  purple: 'M0-1L0.8 0L0 1L-0.8 0Z',
  white: circle(0, 0, 0.92) + circle(0, 0, 0.5),
  black: 'M-0.78-0.78H0.78V0.78H-0.78Z',
  locomotive:
    circle(0, 0, 0.98) +
    circle(0, 0, 0.72) +
    circle(0, 0, 0.24) +
    'M-0.08-0.74H0.08V0.74H-0.08ZM-0.74-0.08H0.74V0.08H-0.74Z',
};

/** Seat colours for the trains, as in the physical game's sets. */
export const SEAT_COLORS = [
  { fill: '#2457a6', light: '#5f8fd6', dark: '#0f2d5e', name: 'Blue' },
  { fill: '#c9302c', light: '#ea6a5f', dark: '#6c1311', name: 'Red' },
  { fill: '#2f8a3e', light: '#66bd6d', dark: '#154a1d', name: 'Green' },
  { fill: '#e2ac1c', light: '#f6d36b', dark: '#7a5a05', name: 'Yellow' },
  { fill: '#2d2d32', light: '#6b6b74', dark: '#09090b', name: 'Black' },
];

export const seatColor = (seat: number) => SEAT_COLORS[seat % SEAT_COLORS.length]!;
