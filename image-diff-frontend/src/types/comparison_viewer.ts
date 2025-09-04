import type { Comparison } from './comparison';

export type ComparisonViewerProps = {
  comparison: Comparison;
  sliderValue: number;
  setSliderValue: (v: number) => void;
};
