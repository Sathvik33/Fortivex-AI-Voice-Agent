import { useState, useEffect, useCallback } from 'react';
import { Search, RefreshCw } from 'lucide-react';
import { api } from '../api';

const CATEGORY_BADGE = {
  HOT: 'bg-rose-50 text-rose-700 border-rose-200',
  WARM: 'bg-amber-50 text-amber-700 border-amber-200',
  COLD: 'bg-sky-50 text-sky-700 border-sky-200',
};

const SENTIMENT_BADGE = {
  positive: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  neutral: 'bg-slate-100 text-slate-600 border-slate-200',
  negative: 'bg-rose-50 text-rose-700 border-rose-200',
};

export default function CustomersView() {
  const [customers, setCustomers] = useState([]);
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('All');
  const [sentiment, setSentiment] = useState('All');
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.getCustomers({ search, category, sentiment });
      setCustomers(data.customers || []);
    } catch {
      setCustomers([]);
    } finally {
      setLoading(false);
    }
  }, [search, category, sentiment]);

  useEffect(() => {
    const t = setTimeout(load, 200);
    return () => clearTimeout(t);
  }, [load]);

  return (
    <div className="space-y-5">
      {/* Filters */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-wrap items-center gap-3 shadow-xs">
        <div className="relative flex-1 min-w-[240px]">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name, email, or company..."
            className="w-full pl-9 pr-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
          />
        </div>

        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none"
        >
          <option value="All">All Categories</option>
          <option value="HOT">HOT</option>
          <option value="WARM">WARM</option>
          <option value="COLD">COLD</option>
        </select>

        <select
          value={sentiment}
          onChange={(e) => setSentiment(e.target.value)}
          className="px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none"
        >
          <option value="All">All Sentiments</option>
          <option value="positive">Positive</option>
          <option value="neutral">Neutral</option>
          <option value="negative">Negative</option>
        </select>

        <button
          onClick={load}
          className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Table + Detail Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className={`${selected ? 'lg:col-span-8' : 'lg:col-span-12'} bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden`}>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50/70 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  <th className="py-3 px-4">Customer</th>
                  <th className="py-3 px-4">Company</th>
                  <th className="py-3 px-4">Score</th>
                  <th className="py-3 px-4">Category</th>
                  <th className="py-3 px-4">Sentiment</th>
                  <th className="py-3 px-4">Budget</th>
                  <th className="py-3 px-4">Last Contact</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                {customers.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-xs text-slate-400">
                      {loading ? 'Loading customers...' : 'No matching customer records found.'}
                    </td>
                  </tr>
                ) : (
                  customers.map((c) => (
                    <tr
                      key={c.id}
                      onClick={() => setSelected(c)}
                      className={`cursor-pointer transition-colors hover:bg-slate-50 ${
                        selected?.id === c.id ? 'bg-blue-50/50' : ''
                      }`}
                    >
                      <td className="py-3 px-4">
                        <div className="font-medium text-slate-900">{c.name}</div>
                        <div className="text-xs text-slate-500">{c.email || '—'}</div>
                      </td>
                      <td className="py-3 px-4 text-slate-700">{c.company || '—'}</td>
                      <td className="py-3 px-4 font-semibold text-slate-900 tabular-nums">{Math.round(c.lead_score)}</td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${CATEGORY_BADGE[c.lead_category] || CATEGORY_BADGE.COLD}`}>
                          {c.lead_category}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-0.5 text-xs font-medium rounded-full border capitalize ${SENTIMENT_BADGE[c.sentiment] || SENTIMENT_BADGE.neutral}`}>
                          {c.sentiment}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-slate-700 tabular-nums">
                        {c.budget ? `${c.budget.toLocaleString()} ${c.budget_currency}` : '—'}
                      </td>
                      <td className="py-3 px-4 text-xs text-slate-500 tabular-nums">{c.last_interaction || '—'}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {selected && (
          <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4 h-fit">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-base font-semibold text-slate-900">{selected.name}</h3>
                <p className="text-xs text-slate-500">{selected.company || 'No company'}</p>
              </div>
              <button onClick={() => setSelected(null)} className="text-xs text-slate-400 hover:text-slate-700 cursor-pointer">
                Close
              </button>
            </div>
            <dl className="space-y-2.5 text-xs border-t border-slate-100 pt-3">
              <div className="flex justify-between"><dt className="text-slate-500">Email</dt><dd className="text-slate-900">{selected.email || '—'}</dd></div>
              <div className="flex justify-between"><dt className="text-slate-500">Phone</dt><dd className="text-slate-900">{selected.phone || '—'}</dd></div>
              <div className="flex justify-between"><dt className="text-slate-500">Company Size</dt><dd className="text-slate-900">{selected.company_size ? `${selected.company_size} employees` : '—'}</dd></div>
              <div className="flex justify-between"><dt className="text-slate-500">Timeline</dt><dd className="text-slate-900">{selected.purchase_timeline || '—'}</dd></div>
              <div className="flex justify-between"><dt className="text-slate-500">Language</dt><dd className="text-slate-900 uppercase">{selected.preferred_language}</dd></div>
              <div className="flex justify-between"><dt className="text-slate-500">Follow-up</dt><dd className="text-slate-900 capitalize">{selected.follow_up_status}</dd></div>
            </dl>
            {selected.requirements && (
              <div className="border-t border-slate-100 pt-3">
                <div className="text-xs font-medium text-slate-500 mb-1">Requirements</div>
                <p className="text-xs text-slate-700 leading-relaxed">{selected.requirements}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
