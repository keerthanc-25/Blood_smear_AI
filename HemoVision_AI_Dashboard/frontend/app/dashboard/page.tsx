'use client';

import { useEffect, useMemo, useState } from 'react';
import AppShell from '../../components/AppShell';
import StatCard from '../../components/StatCard';
import SmearPreview from '../../components/SmearPreview';
import { Confidence, Distribution } from '../../components/Charts';
import {
  ArrowUpRight,
  CheckCircle2,
  Download,
  AlertTriangle,
} from 'lucide-react';
import type { AnalysisResult, Detection } from '../../lib/api';

function MorphologyPanel({
  title,
  subtitle,
  entries,
  averageConfidence,
  emptyMessage,
  colorClass,
}: {
  title: string;
  subtitle: string;
  entries: Record<string, { count: number; percentage: number }>;
  averageConfidence: number;
  emptyMessage: string;
  colorClass: string;
}) {
  const sortedEntries = Object.entries(entries).sort(
    (a, b) => b[1].count - a[1].count,
  );

  return (
    <section className="panel pipeline">
      <div className="panel-head">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
      </div>

      {sortedEntries.length > 0 ? (
        <div className="conf-list">
          {sortedEntries.map(([name, value]) => (
            <div className="conf" key={name}>
              <b>{name}</b>

              <span>
                <i
                  className={colorClass}
                  style={{
                    width: `${Math.max(
                      0,
                      Math.min(100, value.percentage),
                    )}%`,
                  }}
                />
              </span>

              <strong>
                {value.count} ({value.percentage.toFixed(1)}%)
              </strong>
            </div>
          ))}
        </div>
      ) : (
        <p style={{ fontSize: 12, color: '#8993a7' }}>
          {emptyMessage}
        </p>
      )}

      <div className="security" style={{ marginTop: 14 }}>
        <AlertTriangle size={16} />

        <div>
          <b>
            Average morphology confidence:{' '}
            {(averageConfidence * 100).toFixed(1)}%
          </b>

          <p>
            Predictions are model outputs and should be visually
            verified. Confidence is not a measure of clinical
            diagnostic certainty.
          </p>
        </div>
      </div>
    </section>
  );
}

export default function Dashboard() {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(
    null,
  );

  useEffect(() => {
    try {
      const raw = localStorage.getItem('hemovision-analysis');

      if (raw) {
        setAnalysis(JSON.parse(raw) as AnalysisResult);
      }
    } catch (error) {
      console.error('Could not load saved analysis:', error);
    }
  }, []);

  const distribution = useMemo(
    () =>
      analysis
        ? [
            {
              name: 'RBC',
              value: analysis.percentages.RBC,
              count: analysis.counts.RBC,
            },
            {
              name: 'WBC',
              value: analysis.percentages.WBC,
              count: analysis.counts.WBC,
            },
            {
              name: 'Platelet',
              value: analysis.percentages.Platelet,
              count: analysis.counts.Platelet,
            },
          ]
        : [],
    [analysis],
  );

  if (!analysis) {
    return (
      <AppShell>
        <div className="page">
          <div className="heading">
            <div>
              <small>WORKSPACE / DASHBOARD</small>
              <h1>No analysis loaded</h1>
              <p>
                Upload a blood-smear image to generate AI analysis
                results.
              </p>
            </div>

            <a href="/analysis" className="btn primary">
              <ArrowUpRight size={15} />
              New Analysis
            </a>
          </div>
        </div>
      </AppShell>
    );
  }

  const rbcMorphology =
    analysis.rbc_morphology ?? analysis.morphology ?? {};

  const wbcMorphology = analysis.wbc_morphology ?? {};

  const rbcConfidence =
    analysis.rbc_morphology_average_confidence ?? 0;

  const wbcConfidence =
    analysis.wbc_morphology_average_confidence ?? 0;

  // Only WBC detections with morphology predictions are included.
  const wbcDetections: Detection[] = analysis.detections.filter(
    (detection) =>
      detection.class === 'WBC' &&
      typeof detection.morphology === 'string',
  );

  return (
    <AppShell>
      <div className="page">
        <div className="heading">
          <div>
            <small>WORKSPACE / DASHBOARD</small>
            <h1>Analysis Results</h1>
            <p>
              AI-assisted quantitative analysis of peripheral blood
              smear imagery.
            </p>
          </div>

          <div className="actions">
            <button
              className="btn secondary"
              onClick={() => window.print()}
            >
              <Download size={15} />
              Print / Save Report
            </button>

            <a href="/analysis" className="btn primary">
              <ArrowUpRight size={15} />
              New Analysis
            </a>
          </div>
        </div>

        <div className="success">
          <CheckCircle2 size={17} />

          <div>
            <b>Analysis completed successfully</b>
            <span>
              Sample {analysis.sample_id} · processed in{' '}
              {analysis.elapsed_seconds}s
            </span>
          </div>
        </div>

        <div className="stats">
          <StatCard
            label="RBC"
            value={analysis.counts.RBC}
            sub={`${analysis.percentages.RBC.toFixed(1)}% of detected cells`}
            tone="rbc"
          />

          <StatCard
            label="WBC"
            value={analysis.counts.WBC}
            sub={`${analysis.percentages.WBC.toFixed(1)}% of detected cells`}
            tone="wbc"
          />

          <StatCard
            label="Platelet"
            value={analysis.counts.Platelet}
            sub={`${analysis.percentages.Platelet.toFixed(1)}% of detected cells`}
            tone="plt"
          />

          <StatCard
            label="Total Cells"
            value={analysis.total_cells}
            sub="Detected cell population"
            tone="total"
          />
        </div>

        <div className="results">
          <SmearPreview
            image={analysis.annotated_image}
            detections={analysis.detections}
          />

          <div className="stack">
            <Confidence
              values={[
                {
                  name: 'RBC',
                  value:
                    analysis.average_confidence.RBC * 100,
                },
                {
                  name: 'WBC',
                  value:
                    analysis.average_confidence.WBC * 100,
                },
                {
                  name: 'Platelet',
                  value:
                    analysis.average_confidence.Platelet * 100,
                },
              ]}
            />

            <Distribution data={distribution} />
          </div>
        </div>

        <div className="lower">
          <MorphologyPanel
            title="RBC morphology"
            subtitle="Stage 2 classification of detected RBC crops"
            entries={rbcMorphology}
            averageConfidence={rbcConfidence}
            emptyMessage="No RBCs were detected in this sample."
            colorClass="rbc"
          />

          <MorphologyPanel
            title="WBC morphology"
            subtitle="Classification of detected WBC crops into trained morphology classes"
            entries={wbcMorphology}
            averageConfidence={wbcConfidence}
            emptyMessage="No WBC morphology predictions are available for this sample."
            colorClass="wbc"
          />
        </div>

        <section className="panel" style={{ marginTop: 18 }}>
          <div className="panel-head">
            <div>
              <h3>Individual WBC predictions</h3>
              <p>
                Detection confidence and morphology classification
                confidence are reported separately.
              </p>
            </div>
          </div>

          {wbcDetections.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: 12,
                  textAlign: 'left',
                }}
              >
                <thead>
                  <tr>
                    <th style={{ padding: 10 }}>Detection ID</th>
                    <th style={{ padding: 10 }}>Predicted morphology</th>
                    <th style={{ padding: 10 }}>
                      Detection confidence
                    </th>
                    <th style={{ padding: 10 }}>
                      Morphology confidence
                    </th>
                  </tr>
                </thead>

                <tbody>
                  {wbcDetections.map((detection) => {
                    const morphology = detection.morphology ?? '';
                    const suspiciousPlateletLabel = [
                      'Giant Platelet',
                      'Platelet Cluster',
                    ].includes(morphology);

                    return (
                      <tr key={detection.id}>
                        <td style={{ padding: 10 }}>
                          {detection.id}
                        </td>

                        <td style={{ padding: 10 }}>
                          {morphology}

                          {suspiciousPlateletLabel && (
                            <div
                              style={{
                                marginTop: 4,
                                color: '#b45309',
                                fontSize: 11,
                              }}
                            >
                              Verify class assignment
                            </div>
                          )}
                        </td>

                        <td style={{ padding: 10 }}>
                          {(
                            detection.detection_confidence * 100
                          ).toFixed(1)}
                          %
                        </td>

                        <td style={{ padding: 10 }}>
                          {typeof detection.morphology_confidence ===
                          'number'
                            ? `${(
                                detection.morphology_confidence * 100
                              ).toFixed(1)}%`
                            : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ fontSize: 12, color: '#8993a7' }}>
              No individual WBC morphology predictions are available.
            </p>
          )}
        </section>

        <section className="panel insight" style={{ marginTop: 18 }}>
          <span>✓</span>

          <div>
            <small>RESEARCH INSIGHT</small>
            <h3>Quantitative analysis available</h3>

            <p>
              Cell counts, detection confidence, and RBC/WBC
              morphology distributions are available from the model
              inference. Predictions require appropriate validation.
            </p>

            <a href="/reports">
              Open detailed report <ArrowUpRight size={14} />
            </a>
          </div>
        </section>

        <div className="disclaimer">
          <span>
            <b>Research use:</b> This interface presents AI-assisted
            analysis for research and educational purposes. Model
            predictions are not definitive clinical diagnoses.
          </span>
        </div>
      </div>
    </AppShell>
  );
}

