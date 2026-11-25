import { useState, useEffect } from 'react';
import { Upload, Search, FileText, RefreshCw, CheckCircle2, AlertCircle } from 'lucide-react';
import { api } from '../api';

export default function KnowledgeBaseView() {
  const [docs, setDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null);
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState([]);

  const loadDocs = async () => {
    try {
      const data = await api.getDocuments();
      setDocs(data.documents || []);
    } catch {
      setDocs([]);
    }
  };

  useEffect(() => {
    loadDocs();
  }, []);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg(null);
    try {
      const res = await api.uploadDocument(file);
      if (res.status === 'success') {
        setUploadMsg({ type: 'ok', text: `Indexed "${res.filename}" (${res.chunks} chunks in ${res.processing_time})` });
      } else if (res.status === 'duplicate') {
        setUploadMsg({ type: 'ok', text: `"${res.filename}" is already indexed.` });
      } else {
        setUploadMsg({ type: 'err', text: res.error || 'Upload failed' });
      }
      await loadDocs();
    } catch (err) {
      setUploadMsg({ type: 'err', text: err.message });
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    try {
      const res = await api.queryDocuments(query.trim(), 4);
      setResults(res.chunks || []);
    } catch {
      setResults([]);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left: Upload + Semantic Search Tester */}
      <div className="lg:col-span-5 space-y-5">
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <h3 className="text-sm font-semibold text-slate-900 mb-1">Ingest Document</h3>
          <p className="text-xs text-slate-500 mb-4">
            Upload PDF, DOCX, TXT, or Markdown files to embed into PgVector.
          </p>
          <label className="flex flex-col items-center justify-center border-2 border-dashed border-slate-200 hover:border-blue-400 rounded-xl p-6 cursor-pointer transition-colors bg-slate-50/50">
            <Upload className="w-6 h-6 text-slate-400 mb-2" />
            <span className="text-xs font-medium text-slate-700">
              {uploading ? 'Processing & embedding...' : 'Choose a file to upload'}
            </span>
            <span className="text-[11px] text-slate-400 mt-1">PDF, TXT, DOCX, MD</span>
            <input type="file" accept=".pdf,.txt,.docx,.doc,.md" onChange={handleUpload} disabled={uploading} className="hidden" />
          </label>
          {uploadMsg && (
            <div className={`mt-3 p-3 rounded-lg text-xs flex items-center gap-2 ${
              uploadMsg.type === 'ok' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-rose-50 text-rose-700 border border-rose-200'
            }`}>
              {uploadMsg.type === 'ok' ? <CheckCircle2 className="w-4 h-4 shrink-0" /> : <AlertCircle className="w-4 h-4 shrink-0" />}
              <span>{uploadMsg.text}</span>
            </div>
          )}
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <h3 className="text-sm font-semibold text-slate-900 mb-1">Test Semantic Retrieval</h3>
          <p className="text-xs text-slate-500 mb-3">Query PgVector directly to inspect grounded chunks.</p>
          <form onSubmit={handleSearch} className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. What is CRM Pro pricing?"
              className="flex-1 px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
            />
            <button
              type="submit"
              disabled={searching}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 cursor-pointer"
            >
              <Search className="w-3.5 h-3.5" />
              Search
            </button>
          </form>

          {results.length > 0 && (
            <div className="mt-4 space-y-2.5">
              {results.map((r, i) => (
                <div key={i} className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs">
                  <div className="flex items-center justify-between font-medium text-slate-800 mb-1">
                    <span>{r.source}</span>
                    <span className="text-blue-600 tabular-nums">{Math.round(r.similarity * 100)}% match</span>
                  </div>
                  <p className="text-slate-600 leading-relaxed">{r.content}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right: Indexed Documents Table */}
      <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden h-fit">
        <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Indexed Documents</h3>
            <p className="text-xs text-slate-500">{docs.length} document(s) in NeonDB PgVector</p>
          </div>
          <button
            onClick={loadDocs}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 border border-slate-200 rounded-lg hover:bg-slate-50 cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </button>
        </div>

        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50/70 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              <th className="py-3 px-4">Filename</th>
              <th className="py-3 px-4">Type</th>
              <th className="py-3 px-4">Chunks</th>
              <th className="py-3 px-4">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-sm">
            {docs.map((d) => (
              <tr key={d.id} className="hover:bg-slate-50">
                <td className="py-3 px-4 flex items-center gap-2 font-medium text-slate-900">
                  <FileText className="w-4 h-4 text-slate-400" />
                  {d.filename}
                </td>
                <td className="py-3 px-4 text-xs uppercase text-slate-500">{d.file_type}</td>
                <td className="py-3 px-4 tabular-nums text-slate-700">{d.total_chunks}</td>
                <td className="py-3 px-4">
                  <span className={`px-2 py-0.5 text-xs font-medium rounded-full border ${
                    d.status === 'ready'
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : 'bg-amber-50 text-amber-700 border-amber-200'
                  }`}>
                    {d.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
