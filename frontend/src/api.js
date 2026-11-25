const API_BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText || `Request failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  getHealth: () => request('/health'),

  login: ({ name, email }) =>
    request('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email }),
    }),

  sendTextMessage: (payload) =>
    request('/chat/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  sendVoiceMessage: (audioBlob, { sessionId, customerId, conversationHistory }) => {
    const form = new FormData();
    form.append('audio', audioBlob, 'recording.webm');
    if (sessionId) form.append('session_id', sessionId);
    if (customerId) form.append('customer_id', customerId);
    form.append('conversation_history', JSON.stringify(conversationHistory || []));
    return request('/chat/voice', { method: 'POST', body: form });
  },

  summarizeCall: (payload) =>
    request('/chat/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  getCustomers: ({ search = '', category = 'All', sentiment = 'All' } = {}) => {
    const params = new URLSearchParams();
    if (search) params.set('search', search);
    if (category && category !== 'All') params.set('category', category);
    if (sentiment && sentiment !== 'All') params.set('sentiment', sentiment);
    return request(`/customers?${params.toString()}`);
  },

  getDocuments: () => request('/documents'),

  uploadDocument: (file) => {
    const form = new FormData();
    form.append('file', file);
    return request('/documents/upload', { method: 'POST', body: form });
  },

  queryDocuments: (query, topK = 4) =>
    request('/documents/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK }),
    }),

  getAppointments: () => request('/appointments'),
  getAvailableSlots: () => request('/appointments/slots'),
  cancelAppointment: (id) => request(`/appointments/${id}/cancel`, { method: 'POST' }),

  getAnalytics: () => request('/analytics'),
};
