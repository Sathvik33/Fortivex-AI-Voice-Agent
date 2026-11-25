import { useState, useEffect } from 'react';
import { RefreshCw } from 'lucide-react';
import { api } from '../api';

function DistributionBars({ title, labels = [], values = [], colorMap = {} }) {
  const total = values.reduce((a, b) => a + b, 0);
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <h3 className="text-sm font-semibold text-slate-900 mb-4">{title}</h3>
      {total === 0 ? (
        <p className="text-xs text-slate-400 py-8 text-center">No data recorded yet.</p>
      ) : (
        <div className="space-y-3">
          {labels.map((label, i) => {
            const count = values[i] || 0;
            const pct = total > 0 ? Math.round((count / total) * 100) : 0;
            const barColor = colorMap[label] || 'bg-blue-600';
            return (
              <div key={label}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-medium text-slate-700 capitalize">{label}</span>
                  <span className="text-slate-500 tabular-nums">{count} ({pct}%)</span>
                </div>
                <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div className={`h-full ${barColor}`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function AnalyticsView() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const res = await api.getAnalytics();
      setData(res);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const kpis = data?.kpis || {};

  const statCards = [
    { label: 'Total Calls', value: kpis.total_calls ?? 0 },
    { label: 'Unique Customers', value: kpis.total_customers ?? 0 },
    { label: 'HOT Leads', value: kpis.hot_leads ?? 0 },
    { label: 'Confirmed Demos', value: kpis.appointments_confirmed ?? 0 },
    { label: 'Avg Lead Score', value: `${kpis.avg_lead_score ?? 0}/100` },
    { label: 'Conversion Rate', value: `${kpis.appointment_conversion_rate ?? 0}%` },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Pipeline & Call Analytics</h2>
          <p className="text-xs text-slate-500">Live metrics queried directly from NeonDB PostgreSQL</p>
        </div>
        <button
          onClick={load}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* KPI Strip */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {statCards.map((s) => (
          <div key={s.label} className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs">
            <div className="text-xs text-slate-500">{s.label}</div>
            <div className="text-2xl font-bold text-slate-900 mt-1 tabular-nums">{s.value}</div>
          </div>
        ))}
      </div>

      {/* Distributions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <DistributionBars
          title="Lead Category Distribution"
          labels={data?.lead_distribution?.labels}
          values={data?.lead_distribution?.values}
          colorMap={{ HOT: 'bg-rose-500', WARM: 'bg-amber-500', COLD: 'bg-sky-500' }}
        />
        <DistributionBars
          title="Customer Sentiment Distribution"
          labels={data?.sentiment_distribution?.labels}
          values={data?.sentiment_distribution?.values}
          colorMap={{ positive: 'bg-emerald-500', neutral: 'bg-slate-400', negative: 'bg-rose-500' }}
        />
      </div>

      {/* Summaries & Recommendations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <h3 className="text-sm font-semibold text-slate-900 mb-3">Recent Call Summaries</h3>
          {(data?.recent_summaries || []).length === 0 ? (
            <p className="text-xs text-slate-400 py-6 text-center">No call summaries saved yet.</p>
          ) : (
            <div className="space-y-3">
              {data.recent_summaries.map((s) => (
                <div key={s.id} className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 text-xs">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-slate-900">
                      {s.customer_name} {s.company ? `· ${s.company}` : ''}
                    </span>
                    <span className="text-slate-500 tabular-nums">{Math.round(s.lead_score)}/100 ({s.lead_category})</span>
                  </div>
                  <p className="text-slate-600 leading-relaxed">{s.recommended_action}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <h3 className="text-sm font-semibold text-slate-900 mb-3">Follow-Up Queue</h3>
          {(data?.recommendations || []).length === 0 ? (
            <p className="text-xs text-slate-400 py-6 text-center">No pending HOT or WARM follow-ups.</p>
          ) : (
            <div className="space-y-3">
              {data.recommendations.map((r) => (
                <div key={r.id} className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 text-xs flex items-start justify-between gap-3">
                  <div>
                    <div className="font-semibold text-slate-900">
                      {r.name} {r.company ? `· ${r.company}` : ''}
                    </div>
                    <p className="text-slate-600 mt-1">{r.action}</p>
                  </div>
                  <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium border shrink-0 ${
                    r.priority === 'Urgent'
                      ? 'bg-rose-50 text-rose-700 border-rose-200'
                      : 'bg-amber-50 text-amber-700 border-amber-200'
                  }`}>
                    {r.priority}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
