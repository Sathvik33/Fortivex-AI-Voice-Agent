import { useState, useEffect } from 'react';
import { PhoneCall, Users, BookOpen, Calendar, BarChart3, LogOut } from 'lucide-react';
import { api } from './api';
import LoginView from './components/LoginView';
import SalesConsole from './components/SalesConsole';
import CustomersView from './components/CustomersView';
import KnowledgeBaseView from './components/KnowledgeBaseView';
import AppointmentsView from './components/AppointmentsView';
import AnalyticsView from './components/AnalyticsView';

const NAV_ITEMS = [
  { id: 'console', label: 'Sales Console', icon: PhoneCall },
  { id: 'customers', label: 'Customers CRM', icon: Users },
  { id: 'kb', label: 'Knowledge Base', icon: BookOpen },
  { id: 'appointments', label: 'Appointments', icon: Calendar },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
];

export default function App() {
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem('techvex_user');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [activeTab, setActiveTab] = useState('console');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api.getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const handleLogin = (loggedInUser) => {
    setUser(loggedInUser);
    try {
      localStorage.setItem('techvex_user', JSON.stringify(loggedInUser));
    } catch {
      // ignore
    }
  };

  const handleLogout = () => {
    setUser(null);
    try {
      localStorage.removeItem('techvex_user');
    } catch {
      // ignore
    }
  };

  if (!user) {
    return <LoginView onLogin={handleLogin} />;
  }

  const currentNav = NAV_ITEMS.find((n) => n.id === activeTab);

  return (
    <div className="h-screen overflow-hidden flex bg-slate-50 text-slate-900">
      {/* Sidebar */}
      <aside className="w-60 bg-white border-r border-slate-200 flex flex-col shrink-0">
        <div className="h-14 px-5 border-b border-slate-200 flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-slate-900 text-white flex items-center justify-center text-xs font-bold">
            TV
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-900 leading-none">TechVex</div>
            <div className="text-[11px] text-slate-500 mt-0.5">Sales Voice Agent</div>
          </div>
        </div>

        <nav className="flex-1 p-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium transition-colors cursor-pointer ${
                  active
                    ? 'bg-slate-900 text-white'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <Icon className="w-4 h-4" />
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* System Health */}
        <div className="p-4 border-t border-slate-200 space-y-1.5">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">
            Services
          </div>
          {['database', 'ollama', 'stt', 'tts', 'rag'].map((svc) => {
            const st = health?.[svc]?.status;
            const ok = st === 'healthy' || st === 'ready';
            return (
              <div key={svc} className="flex items-center justify-between text-xs">
                <span className="text-slate-600 uppercase">{svc}</span>
                <span className="inline-flex items-center gap-1.5 text-[11px] text-slate-500">
                  <span className={`w-1.5 h-1.5 rounded-full ${ok ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                  {st || 'checking'}
                </span>
              </div>
            );
          })}
        </div>

        {/* Signed-in User */}
        <div className="p-4 border-t border-slate-200 flex items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="text-xs font-semibold text-slate-900 truncate">{user.name}</div>
            <div className="text-[11px] text-slate-500 truncate">{user.email || 'No email'}</div>
          </div>
          <button
            onClick={handleLogout}
            title="Sign out"
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 cursor-pointer shrink-0"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
        <header className="h-14 bg-white border-b border-slate-200 px-6 flex items-center justify-between shrink-0">
          <h1 className="text-sm font-semibold text-slate-900">{currentNav?.label}</h1>
          <div className="text-xs text-slate-500">
            Signed in as <span className="font-medium text-slate-800">{user.name}</span>
          </div>
        </header>

        <main className={`flex-1 p-6 min-h-0 ${activeTab === 'console' ? 'overflow-hidden' : 'overflow-y-auto'}`}>
          {activeTab === 'console' && <SalesConsole user={user} />}
          {activeTab === 'customers' && <CustomersView />}
          {activeTab === 'kb' && <KnowledgeBaseView />}
          {activeTab === 'appointments' && <AppointmentsView />}
          {activeTab === 'analytics' && <AnalyticsView />}
        </main>
      </div>
    </div>
  );
}
