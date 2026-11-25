import { useState, useEffect } from 'react';
import { Calendar, Clock, RefreshCw, XCircle } from 'lucide-react';
import { api } from '../api';

export default function AppointmentsView() {
  const [appointments, setAppointments] = useState([]);
  const [slots, setSlots] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const [aRes, sRes] = await Promise.all([api.getAppointments(), api.getAvailableSlots()]);
      setAppointments(aRes.appointments || []);
      setSlots(sRes.slots || []);
    } catch {
      setAppointments([]);
      setSlots([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleCancel = async (id) => {
    try {
      await api.cancelAppointment(id);
      await load();
    } catch {
      // ignore
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Booked Appointments */}
      <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden h-fit">
        <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Scheduled Appointments</h3>
            <p className="text-xs text-slate-500">Confirmed via explicit customer consent</p>
          </div>
          <button
            onClick={load}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 border border-slate-200 rounded-lg hover:bg-slate-50 cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50/70 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              <th className="py-3 px-4">Customer</th>
              <th className="py-3 px-4">Date & Time</th>
              <th className="py-3 px-4">Purpose</th>
              <th className="py-3 px-4">Status</th>
              <th className="py-3 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-sm">
            {appointments.length === 0 ? (
              <tr>
                <td colSpan={5} className="py-12 text-center text-xs text-slate-400">
                  No appointments booked yet.
                </td>
              </tr>
            ) : (
              appointments.map((a) => (
                <tr key={a.id} className="hover:bg-slate-50">
                  <td className="py-3 px-4">
                    <div className="font-medium text-slate-900">{a.customer_name}</div>
                    <div className="text-xs text-slate-500">{a.company || '—'}</div>
                  </td>
                  <td className="py-3 px-4 text-xs text-slate-700 tabular-nums">
                    <div>{a.appointment_date}</div>
                    <div className="text-slate-500">{a.appointment_time}</div>
                  </td>
                  <td className="py-3 px-4 text-xs text-slate-700">{a.purpose}</td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-0.5 text-xs font-medium rounded-full border capitalize ${
                      a.status === 'confirmed'
                        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                        : 'bg-slate-100 text-slate-500 border-slate-200'
                    }`}>
                      {a.status}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right">
                    {a.status === 'confirmed' && (
                      <button
                        onClick={() => handleCancel(a.id)}
                        className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-rose-700 hover:bg-rose-50 rounded-md cursor-pointer"
                      >
                        <XCircle className="w-3.5 h-3.5" />
                        Cancel
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Upcoming Available Slots */}
      <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 shadow-xs h-fit">
        <div className="flex items-center gap-2 mb-1">
          <Calendar className="w-4 h-4 text-slate-400" />
          <h3 className="text-sm font-semibold text-slate-900">Available Slots (Next 14 Days)</h3>
        </div>
        <p className="text-xs text-slate-500 mb-4">Protected by SQL row locking against double-booking.</p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-h-[480px] overflow-y-auto pr-1">
          {slots.map((s) => (
            <div
              key={s.id}
              className="p-3 rounded-lg border border-slate-200 bg-slate-50/50 flex items-center justify-between text-xs"
            >
              <div>
                <div className="font-medium text-slate-900">{s.day_name}</div>
                <div className="text-slate-500 tabular-nums">{s.date}</div>
              </div>
              <div className="inline-flex items-center gap-1 px-2 py-1 rounded bg-white border border-slate-200 font-medium text-slate-700 tabular-nums">
                <Clock className="w-3 h-3 text-slate-400" />
                {s.time}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
