import { Slider } from '@mui/material';
import type { ComparisonViewerProps } from './types/comparison_viewer';

export default function ComparisonViewer({
  comparison,
  sliderValue,
  setSliderValue,
}: ComparisonViewerProps) {
  const thresholds: number[] = [];
  // 5 - 65
  for (const key of Object.keys(comparison.diffImages)) {
    thresholds.push(Number(key));
  }
  // Sort from lowest to highest, if a-b returns positive, a comes after b
  thresholds.sort((a, b) => a - b);

  // Second div: Grid of 3 equal width columns with 1 rem (16px) gap between columns
  // max-w-md mx-auto -> Sets 28 rem max witdth and automates the left and right margin widths
  return (
    <div className="mt-6">
      {/* Image panels */}
      <div className="grid grid-cols-3 gap-4">
        <img
          src={comparison.beforeUrl}
          alt="Before"
          className="w-full h-auto max-h-[400px] object-contain border rounded-lg shadow"
        />
        <img
          src={comparison.afterUrl}
          alt="After"
          className="w-full h-auto max-h-[400px] object-contain border rounded-lg shadow"
        />
        <div className="flex flex-col items-center">
          <img
            src={comparison.diffImages[sliderValue]}
            alt={`Diff at threshold ${sliderValue}`}
            className="w-full h-auto max-h-[400px] object-contain border rounded-lg shadow"
          />
          <Slider
            value={sliderValue}
            onChange={(_, v) => setSliderValue(v as number)}
            min={Math.min(...thresholds)}
            max={Math.max(...thresholds)}
            step={5}
            marks={thresholds.map((t) => ({ value: t, label: `${t}` }))}
            sx={{ width: '90%', marginTop: 2 }}
          />
        </div>
      </div>

      {/* Scores panel */}
      <div className="mt-4 p-4 bg-white rounded-lg shadow max-w-md mx-auto">
        <h3 className="text-lg font-semibold mb-2">Scores</h3>
        <p>Correlation: {comparison.scores.correlation.toFixed(2)}%</p>
        <p>Chi-Square: {comparison.scores.chiSquare.toFixed(2)}</p>
        <p>Bhattacharyya: {comparison.scores.bhattacharyya.toFixed(2)}%</p>
      </div>
    </div>
  );
}
