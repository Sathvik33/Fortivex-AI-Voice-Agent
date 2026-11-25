import { useState, useRef, useEffect } from 'react';
import {
  Phone,
  PhoneOff,
  Mic,
  Square,
  Send,
  Volume2,
  VolumeX,
  FileText,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Loader2,
  CalendarCheck,
  User,
} from 'lucide-react';
import { api } from '../api';

function base64ToAudioUrl(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: 'audio/wav' });
  return URL.createObjectURL(blob);
}

const CATEGORY_STYLES = {
  HOT: 'bg-rose-50 text-rose-700 border-rose-200',
  WARM: 'bg-amber-50 text-amber-700 border-amber-200',
  COLD: 'bg-sky-50 text-sky-700 border-sky-200',
};

const SENTIMENT_STYLES = {
  positive: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  neutral: 'bg-slate-100 text-slate-700 border-slate-200',
  negative: 'bg-rose-50 text-rose-700 border-rose-200',
};

const CALL_STATUS_META = {
  idle: { label: 'Idle', dot: 'bg-slate-400' },
  listening: { label: 'Listening — speak naturally', dot: 'bg-emerald-500 animate-pulse' },
  speaking_user: { label: 'Capturing speech...', dot: 'bg-blue-600 animate-pulse' },
  thinking: { label: 'Agent thinking...', dot: 'bg-amber-500 animate-pulse' },
  speaking_agent: { label: 'Agent speaking...', dot: 'bg-violet-600 animate-pulse' },
};

export default function SalesConsole({ user }) {
  const initialGreeting = user?.greeting || (user?.name ? `Hello ${user.name}! How can I help you today?` : null);
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [customerId, setCustomerId] = useState(() => user?.customer_id || null);
  const [history, setHistory] = useState(() =>
    initialGreeting ? [{ role: 'assistant', content: initialGreeting }] : []
  );
  const [textInput, setTextInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  // Manual push-to-talk state
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);

  // Hands-free Live Call state
  const [isLiveCall, setIsLiveCall] = useState(false);
  const [callStatus, setCallStatus] = useState('idle');
  const [micLevel, setMicLevel] = useState(0);

  const [error, setError] = useState('');
  const [leadScore, setLeadScore] = useState(0);
  const [leadCategory, setLeadCategory] = useState('COLD');
  const [sentiment, setSentiment] = useState({ overall: 'neutral' });
  const [customerContext, setCustomerContext] = useState(() => ({
    ...(user?.name ? { name: user.name } : {}),
    ...(user?.email ? { email: user.email } : {}),
  }));
  const [qualData, setQualData] = useState({});
  const [retrievedDocs, setRetrievedDocs] = useState([]);
  const [bookingRequest, setBookingRequest] = useState({});
  const [summary, setSummary] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);

  // Refs for live-call loop (avoid stale closures)
  const sessionIdRef = useRef(sessionId);
  const customerIdRef = useRef(customerId);
  const historyRef = useRef(history);
  const isLiveCallRef = useRef(false);
  const callStatusRef = useRef('idle');

  const streamRef = useRef(null);
  const audioCtxRef = useRef(null);
  const rafRef = useRef(null);
  const utteranceRecorderRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const chatEndRef = useRef(null);
  const audioPlayerRef = useRef(null);
  const activeAudioRef = useRef(null);
  const audioUrlRef = useRef(null);

  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
  useEffect(() => { customerIdRef.current = customerId; }, [customerId]);
  useEffect(() => { historyRef.current = history; }, [history]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, isProcessing, callStatus]);

  useEffect(() => {
    return () => {
      stopLiveCall();
      if (timerRef.current) clearInterval(timerRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (activeAudioRef.current) {
        activeAudioRef.current.pause();
        activeAudioRef.current = null;
      }
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, []);

  const setStatusBoth = (st) => {
    callStatusRef.current = st;
    setCallStatus(st);
  };

  const applyAgentResult = (res) => {
    if (res.customer_id) setCustomerId(res.customer_id);
    if (Array.isArray(res.conversation_history)) setHistory(res.conversation_history);
    if (typeof res.lead_score === 'number') setLeadScore(res.lead_score);
    if (res.lead_category) setLeadCategory(res.lead_category);
    if (res.sentiment) setSentiment(res.sentiment);
    if (res.customer_context) setCustomerContext(res.customer_context);
    if (res.qualification_data) setQualData(res.qualification_data);
    if (Array.isArray(res.retrieved_documents)) setRetrievedDocs(res.retrieved_documents);
    if (res.booking_request) setBookingRequest(res.booking_request);
  };

  const playAudioAndResume = (b64) => {
    if (!b64) {
      if (isLiveCallRef.current) setStatusBoth('listening');
      else setStatusBoth('idle');
      return;
    }

    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current.onended = null;
      activeAudioRef.current.onerror = null;
      activeAudioRef.current = null;
    }

    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
    }
    const url = base64ToAudioUrl(b64);
    audioUrlRef.current = url;
    setAudioUrl(url);

    // Use a dedicated Audio instance detached from React DOM re-renders
    const audio = new Audio(url);
    activeAudioRef.current = audio;

    if (isLiveCallRef.current) setStatusBoth('speaking_agent');

    const finishPlayback = () => {
      if (activeAudioRef.current === audio) {
        activeAudioRef.current = null;
      }
      if (isLiveCallRef.current) {
        setStatusBoth('listening');
      } else {
        setStatusBoth('idle');
      }
    };

    audio.onended = finishPlayback;
    audio.onerror = finishPlayback;
    audio.play().catch((err) => {
      console.warn('Audio autoplay blocked:', err);
      finishPlayback();
    });
  };

  const interruptAgentSpeech = () => {
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current.currentTime = 0;
      activeAudioRef.current = null;
    }
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      audioPlayerRef.current.currentTime = 0;
    }
    if (isLiveCallRef.current) {
      setStatusBoth('listening');
    } else {
      setStatusBoth('idle');
    }
  };

  // ─── HANDS-FREE LIVE CALL (AUTO VAD -> STT -> AGENT -> TTS -> LISTEN) ───────

  const startLiveCall = async () => {
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      audioCtxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.fftSize);

      isLiveCallRef.current = true;
      setIsLiveCall(true);
      setStatusBoth('listening');

      const SPEECH_THRESHOLD = 0.025;
      const SILENCE_MS = 1200;
      const MIN_SPEECH_MS = 350;

      let utteranceChunks = [];
      let speechStartTime = 0;
      let lastSpeechTime = 0;

      const startUtteranceCapture = () => {
        if (!isLiveCallRef.current || utteranceRecorderRef.current) return;
        const rec = new MediaRecorder(stream);
        utteranceChunks = [];
        rec.ondataavailable = (ev) => {
          if (ev.data.size > 0) utteranceChunks.push(ev.data);
        };
        rec.onstop = async () => {
          const duration = Date.now() - speechStartTime;
          const blob = new Blob(utteranceChunks, { type: rec.mimeType || 'audio/webm' });
          utteranceRecorderRef.current = null;

          if (!isLiveCallRef.current) return;
          if (duration < MIN_SPEECH_MS || blob.size < 800) {
            setStatusBoth('listening');
            return;
          }

          setStatusBoth('thinking');
          setIsProcessing(true);
          try {
            const res = await api.sendVoiceMessage(blob, {
              sessionId: sessionIdRef.current,
              customerId: customerIdRef.current,
              conversationHistory: historyRef.current,
            });
            if (res.empty_audio) {
              if (isLiveCallRef.current) setStatusBoth('listening');
            } else {
              applyAgentResult(res);
              playAudioAndResume(res.audio_base64);
            }
          } catch (err) {
            setError(err.message || 'Live call turn failed.');
            if (isLiveCallRef.current) setStatusBoth('listening');
          } finally {
            setIsProcessing(false);
          }
        };

        utteranceRecorderRef.current = rec;
        speechStartTime = Date.now();
        lastSpeechTime = Date.now();
        rec.start();
        setStatusBoth('speaking_user');
      };

      const monitor = () => {
        if (!isLiveCallRef.current) return;

        analyser.getByteTimeDomainData(dataArray);
        let sumSq = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const v = (dataArray[i] - 128) / 128;
          sumSq += v * v;
        }
        const rms = Math.sqrt(sumSq / dataArray.length);
        setMicLevel(Math.min(1, rms * 8));

        const st = callStatusRef.current;

        if (st === 'listening' || st === 'speaking_user') {
          if (rms > SPEECH_THRESHOLD) {
            lastSpeechTime = Date.now();
            if (st === 'listening') {
              startUtteranceCapture();
            }
          } else if (st === 'speaking_user' && utteranceRecorderRef.current && utteranceRecorderRef.current.state === 'recording') {
            if (Date.now() - lastSpeechTime > SILENCE_MS) {
              utteranceRecorderRef.current.stop();
            }
          }
        }

        rafRef.current = requestAnimationFrame(monitor);
      };

      rafRef.current = requestAnimationFrame(monitor);
    } catch {
      setError('Microphone permission denied or unavailable.');
      stopLiveCall();
    }
  };

  const stopLiveCall = () => {
    isLiveCallRef.current = false;
    setIsLiveCall(false);
    setStatusBoth('idle');
    setMicLevel(0);

    if (utteranceRecorderRef.current && utteranceRecorderRef.current.state !== 'inactive') {
      utteranceRecorderRef.current.stop();
    }
    utteranceRecorderRef.current = null;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (activeAudioRef.current) {
      activeAudioRef.current.pause();
      activeAudioRef.current = null;
    }
    const el = audioPlayerRef.current;
    if (el) {
      el.pause();
    }
  };

  // ─── MANUAL TEXT / PUSH-TO-TALK ─────────────────────────────────────────────

  const handleSendText = async (e) => {
    e?.preventDefault();
    const msg = textInput.trim();
    if (!msg || isProcessing) return;

    setError('');
    setTextInput('');
    setHistory((prev) => [...prev, { role: 'user', content: msg }]);
    setIsProcessing(true);

    try {
      const res = await api.sendTextMessage({
        session_id: sessionId,
        customer_id: customerId,
        message: msg,
        conversation_history: history,
        include_tts: true,
      });
      applyAgentResult(res);
      playAudioAndResume(res.audio_base64);
    } catch (err) {
      setError(err.message || 'Failed to get agent response.');
    } finally {
      setIsProcessing(false);
    }
  };

  const startManualRecording = async () => {
    setError('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      recorder.ondataavailable = (ev) => {
        if (ev.data.size > 0) audioChunksRef.current.push(ev.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        if (blob.size < 500) return;

        setIsProcessing(true);
        try {
          const res = await api.sendVoiceMessage(blob, {
            sessionId,
            customerId,
            conversationHistory: history,
          });
          if (res.empty_audio) {
            setError('No clear speech detected. Please speak closer to the microphone.');
          } else {
            applyAgentResult(res);
            playAudioAndResume(res.audio_base64);
          }
        } catch (err) {
          setError(err.message || 'Voice processing failed.');
        } finally {
          setIsProcessing(false);
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
      setRecordSeconds(0);
      timerRef.current = setInterval(() => setRecordSeconds((s) => s + 1), 1000);
    } catch {
      setError('Microphone permission denied or unavailable.');
    }
  };

  const stopManualRecording = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const handleSummarize = async () => {
    if (history.length === 0 || isProcessing) return;
    if (isLiveCall) stopLiveCall();
    setIsProcessing(true);
    setError('');
    try {
      const res = await api.summarizeCall({
        session_id: sessionId,
        customer_id: customerId,
        conversation_history: history,
        customer_context: customerContext,
        lead_score: leadScore,
        lead_category: leadCategory,
        sentiment,
        booking_request: bookingRequest,
      });
      setSummary(res.summary);
    } catch (err) {
      setError(err.message || 'Could not generate call summary.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    stopLiveCall();
    setSessionId(crypto.randomUUID());
    setCustomerId(user?.customer_id || null);
    setHistory(initialGreeting ? [{ role: 'assistant', content: initialGreeting }] : []);
    setLeadScore(0);
    setLeadCategory('COLD');
    setSentiment({ overall: 'neutral' });
    setCustomerContext({
      ...(user?.name ? { name: user.name } : {}),
      ...(user?.email ? { email: user.email } : {}),
    });
    setQualData({});
    setRetrievedDocs([]);
    setBookingRequest({});
    setSummary(null);
    setAudioUrl(null);
    setError('');
  };

  const contextEntries = [
    ['Name', customerContext.name],
    ['Company', customerContext.company],
    ['Company Size', customerContext.company_size ? `${customerContext.company_size} employees` : null],
    [
      'Budget',
      customerContext.budget
        ? `${Number(customerContext.budget).toLocaleString()} ${customerContext.budget_currency || 'INR'}`
        : null,
    ],
    ['Timeline', customerContext.purchase_timeline],
    ['Decision Maker', customerContext.is_decision_maker === true ? 'Yes' : customerContext.is_decision_maker === false ? 'No' : null],
    ['Requirements', customerContext.requirements],
  ].filter(([, v]) => v);

  const statusMeta = CALL_STATUS_META[callStatus] || CALL_STATUS_META.idle;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full min-h-0">
      {/* Left: Conversation Stream + Live Call Controls */}
      <div className="lg:col-span-8 flex flex-col bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden h-full min-h-0">
        {/* Toolbar */}
        <div className="px-5 py-3.5 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3 bg-white">
          <div className="flex items-center gap-3">
            {!isLiveCall ? (
              <button
                onClick={startLiveCall}
                disabled={isProcessing || isRecording}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold transition-colors disabled:opacity-40 cursor-pointer"
              >
                <Phone className="w-3.5 h-3.5" />
                Start Live Call
              </button>
            ) : (
              <button
                onClick={stopLiveCall}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold transition-colors cursor-pointer"
              >
                <PhoneOff className="w-3.5 h-3.5" />
                End Live Call
              </button>
            )}

            <div className="flex items-center gap-2 text-xs text-slate-600">
              <span className={`w-2 h-2 rounded-full ${statusMeta.dot}`} />
              <span className="font-medium">{statusMeta.label}</span>
            </div>

            {isLiveCall && (
              <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden" title="Mic level">
                <div
                  className="h-full bg-emerald-500 transition-all duration-75"
                  style={{ width: `${Math.round(micLevel * 100)}%` }}
                />
              </div>
            )}

            {callStatus === 'speaking_agent' && (
              <button
                onClick={interruptAgentSpeech}
                className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium text-amber-700 bg-amber-50 border border-amber-200 rounded-md hover:bg-amber-100 cursor-pointer"
              >
                <VolumeX className="w-3 h-3" />
                Interrupt
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleSummarize}
              disabled={history.length === 0 || isProcessing}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-40 cursor-pointer"
            >
              <FileText className="w-3.5 h-3.5" />
              End & Summarize
            </button>
            <button
              onClick={handleReset}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              New Call
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-slate-50/50">
          {history.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto py-16">
              <div className="w-11 h-11 rounded-full bg-emerald-50 border border-emerald-100 flex items-center justify-center mb-3">
                <Phone className="w-5 h-5 text-emerald-600" />
              </div>
              <h3 className="text-sm font-semibold text-slate-900">Hands-free Live Call Ready</h3>
              <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                Click <strong>Start Live Call</strong> to speak naturally — silence detection automatically triggers STT, LangGraph reasoning, RAG/CRM updates, and TTS audio playback in a continuous loop.
              </p>
            </div>
          ) : (
            history.map((msg, idx) => {
              const isUser = msg.role === 'user';
              return (
                <div key={idx} className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
                  <div
                    className={`max-w-[78%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                      isUser
                        ? 'bg-slate-900 text-white'
                        : 'bg-white border border-slate-200 text-slate-800 shadow-2xs'
                    }`}
                  >
                    <div className={`text-[11px] font-medium mb-1 ${isUser ? 'text-slate-300' : 'text-slate-400'}`}>
                      {isUser ? 'Customer' : 'Sales Agent'}
                    </div>
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  </div>
                </div>
              );
            })
          )}

          {isProcessing && (
            <div className="flex justify-start">
              <div className="bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-xs text-slate-500 flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-600" />
                Transcribing, reasoning & synthesizing speech...
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Audio Playback Bar */}
        <div className={`px-5 py-2 border-t border-slate-200 bg-slate-50 flex items-center gap-3 ${audioUrl ? '' : 'hidden'}`}>
          <Volume2 className="w-4 h-4 text-blue-600 shrink-0" />
          <span className="text-xs font-medium text-slate-600">Agent Audio</span>
          <audio ref={audioPlayerRef} controls src={audioUrl || undefined} className="h-8 flex-1" />
        </div>

        {/* Error Banner */}
        {error && (
          <div className="px-5 py-2 bg-rose-50 border-t border-rose-200 flex items-center gap-2 text-xs text-rose-700">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Fallback Manual Input Bar */}
        <form onSubmit={handleSendText} className="p-4 border-t border-slate-200 bg-white flex items-center gap-2.5">
          {!isRecording ? (
            <button
              type="button"
              onClick={startManualRecording}
              disabled={isProcessing || isLiveCall}
              title="Push-to-talk single turn"
              className="inline-flex items-center gap-1.5 px-3 py-2.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium transition-colors disabled:opacity-40 cursor-pointer shrink-0"
            >
              <Mic className="w-4 h-4" />
              Push-to-Talk
            </button>
          ) : (
            <button
              type="button"
              onClick={stopManualRecording}
              className="inline-flex items-center gap-1.5 px-3 py-2.5 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-xs font-medium transition-colors cursor-pointer shrink-0"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              Stop ({recordSeconds}s)
            </button>
          )}

          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Or type a message..."
            disabled={isProcessing || isRecording}
            className="flex-1 px-3.5 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
          />

          <button
            type="submit"
            disabled={!textInput.trim() || isProcessing || isRecording}
            className="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-white text-xs font-medium transition-colors disabled:opacity-40 cursor-pointer shrink-0"
          >
            <Send className="w-3.5 h-3.5" />
            Send
          </button>
        </form>
      </div>

      {/* Right: Live Inspector */}
      <div className="lg:col-span-4 space-y-4 overflow-y-auto min-h-0 pr-1">
        {/* Qualification Card */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Lead Qualification</span>
            <span className={`px-2.5 py-0.5 text-xs font-semibold rounded-full border ${CATEGORY_STYLES[leadCategory] || CATEGORY_STYLES.COLD}`}>
              {leadCategory}
            </span>
          </div>
          <div className="flex items-baseline gap-1.5 mb-2">
            <span className="text-3xl font-bold text-slate-900 tabular-nums">{Math.round(leadScore)}</span>
            <span className="text-sm text-slate-400">/ 100</span>
          </div>
          <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden mb-3">
            <div
              className="h-full bg-blue-600 transition-all duration-300"
              style={{ width: `${Math.min(100, Math.max(0, leadScore))}%` }}
            />
          </div>
          {qualData.total_score !== undefined && (
            <div className="grid grid-cols-2 gap-2 text-xs text-slate-600 pt-2 border-t border-slate-100">
              <div>Budget: <span className="font-medium text-slate-900 tabular-nums">{qualData.budget_score ?? 0}/25</span></div>
              <div>Timeline: <span className="font-medium text-slate-900 tabular-nums">{qualData.timeline_score ?? 0}/20</span></div>
              <div>Authority: <span className="font-medium text-slate-900 tabular-nums">{qualData.authority_score ?? 0}/20</span></div>
              <div>Fit: <span className="font-medium text-slate-900 tabular-nums">{qualData.requirement_score ?? 0}/20</span></div>
            </div>
          )}
        </div>

        {/* Sentiment & Booking */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Sentiment</span>
            <span className={`px-2.5 py-0.5 text-xs font-medium rounded-full border capitalize ${SENTIMENT_STYLES[sentiment.overall] || SENTIMENT_STYLES.neutral}`}>
              {sentiment.overall || 'neutral'}
            </span>
          </div>
          <div className="flex items-center justify-between text-xs text-slate-600">
            <span>Buying Intent</span>
            <span className="font-medium text-slate-900 capitalize">{sentiment.buying_intent || 'low'}</span>
          </div>
          {bookingRequest?.appointment && (
            <div className="mt-2 p-3 rounded-lg bg-emerald-50 border border-emerald-200 flex items-start gap-2.5">
              <CalendarCheck className="w-4 h-4 text-emerald-600 mt-0.5 shrink-0" />
              <div className="text-xs">
                <div className="font-semibold text-emerald-900">Appointment Confirmed</div>
                <div className="text-emerald-700 mt-0.5">{bookingRequest.appointment.datetime_display}</div>
                {bookingRequest.email_status?.sent && (
                  <div className="text-emerald-700 mt-1">Email sent to {bookingRequest.email_status.to}</div>
                )}
                {bookingRequest.email_status && !bookingRequest.email_status.sent && bookingRequest.email_status.status === 'email_not_configured' && (
                  <div className="text-amber-700 mt-1">Add RESEND_API_KEY or BREVO_API_KEY in .env to send email.</div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Customer Context */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
          <div className="flex items-center gap-2 mb-3">
            <User className="w-4 h-4 text-slate-400" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Customer Memory</span>
          </div>
          {contextEntries.length === 0 ? (
            <p className="text-xs text-slate-400">Extracted details will appear here during the call.</p>
          ) : (
            <dl className="space-y-2 text-xs">
              {contextEntries.map(([label, val]) => (
                <div key={label} className="flex justify-between gap-4">
                  <dt className="text-slate-500 shrink-0">{label}</dt>
                  <dd className="text-slate-900 font-medium text-right">{String(val)}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>

        {/* Retrieved RAG Sources */}
        {retrievedDocs.length > 0 && (
          <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2.5">
              Retrieved Sources
            </div>
            <div className="space-y-2">
              {retrievedDocs.map((doc, idx) => (
                <div key={idx} className="text-xs p-2.5 rounded-lg bg-slate-50 border border-slate-200">
                  <div className="flex items-center justify-between font-medium text-slate-800">
                    <span>{doc.source}</span>
                    <span className="text-blue-600 tabular-nums">{Math.round((doc.similarity || 0) * 100)}%</span>
                  </div>
                  <p className="text-slate-500 mt-1 line-clamp-2">{doc.content}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Call Summary */}
        {summary && (
          <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-emerald-700 mb-2">
              <CheckCircle2 className="w-4 h-4" />
              Call Summary Saved
            </div>
            <p className="text-xs text-slate-700 leading-relaxed">{summary.full_summary}</p>
            <div className="mt-3 pt-3 border-t border-slate-100 text-xs">
              <span className="font-semibold text-slate-900">Next Action: </span>
              <span className="text-slate-600">{summary.recommended_action}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
