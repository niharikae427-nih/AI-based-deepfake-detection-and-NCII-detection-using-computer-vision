// ==========================================================================
// SENTRA AI — Global JS utilities
// ==========================================================================

// ---------------- Toast auto-dismiss ----------------
document.addEventListener('DOMContentLoaded', () => {
  const toasts = document.querySelectorAll('#toast-stack .toast');
  toasts.forEach((t, i) => {
    setTimeout(() => {
      t.style.transition = 'opacity .4s ease, transform .4s ease';
      t.style.opacity = '0';
      t.style.transform = 'translateX(20px)';
      setTimeout(() => t.remove(), 400);
    }, 5000 + i * 400);
  });
});

// ---------------- Ripple effect on .ripple buttons ----------------
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.ripple');
  if (!btn) return;
  const rect = btn.getBoundingClientRect();
  const circle = document.createElement('span');
  const size = Math.max(rect.width, rect.height);
  circle.className = 'ripple-circle';
  circle.style.width = circle.style.height = size + 'px';
  circle.style.left = (e.clientX - rect.left - size / 2) + 'px';
  circle.style.top = (e.clientY - rect.top - size / 2) + 'px';
  btn.appendChild(circle);
  setTimeout(() => circle.remove(), 650);
});

// ---------------- Mobile nav toggle ----------------
function toggleMobileNav() {
  const nav = document.getElementById('mobile-nav');
  if (nav) nav.classList.toggle('hidden');
}

// ---------------- Sidebar toggle (mobile dashboard) ----------------
function toggleSidebar() {
  const sidebar = document.getElementById('app-sidebar');
  if (sidebar) sidebar.classList.toggle('-translate-x-full');
}

// ---------------- Animated stat counters ----------------
function animateCounters() {
  document.querySelectorAll('[data-counter]').forEach((el) => {
    const target = parseFloat(el.getAttribute('data-counter'));
    const decimals = el.getAttribute('data-decimals') ? parseInt(el.getAttribute('data-decimals')) : 0;
    const duration = 1800;
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = target * eased;
      el.textContent = value.toLocaleString(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: decimals });
      if (progress < 1) requestAnimationFrame(tick);
      else el.textContent = target.toLocaleString(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: decimals });
    }
    requestAnimationFrame(tick);
  });
}

// Trigger counters when visible
const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      animateCounters();
      counterObserver.disconnect();
    }
  });
}, { threshold: 0.3 });

document.addEventListener('DOMContentLoaded', () => {
  const statsSection = document.getElementById('stats-section');
  if (statsSection) counterObserver.observe(statsSection);
});

// ---------------- Generic Dropzone factory ----------------
function initDropzone({ zoneId, inputId, previewId, onFile }) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  ['dragenter', 'dragover'].forEach(evt =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add('dragover'); })
  );
  ['dragleave', 'drop'].forEach(evt =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove('dragover'); })
  );
  zone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) { input.files = e.dataTransfer.files; onFile(file); }
  });
  input.addEventListener('change', () => {
    if (input.files[0]) onFile(input.files[0]);
  });
}

// ==========================================================================
// AI Assistant Widget (floating chat)
// ==========================================================================
function toggleAssistant() {
  const panel = document.getElementById('assistant-panel');
  const launcher = document.getElementById('assistant-launcher');
  if (!panel) return;
  panel.classList.toggle('hidden');
  panel.classList.toggle('flex');
  if (!panel.classList.contains('hidden') && launcher) {
    launcher.classList.add('scale-0');
  } else if (launcher) {
    launcher.classList.remove('scale-0');
  }
}

function appendAssistantMessage(text, from) {
  const log = document.getElementById('assistant-log');
  if (!log) return;
  const wrap = document.createElement('div');
  wrap.className = from === 'user'
    ? 'flex justify-end animate-fade-in'
    : 'flex justify-start animate-fade-in';

  const bubble = document.createElement('div');
  bubble.className = from === 'user'
    ? 'max-w-[80%] bg-gradient-to-br from-primary to-accent text-[#04101a] text-sm rounded-2xl rounded-br-sm px-4 py-2.5 font-medium'
    : 'max-w-[80%] bg-white/5 border border-white/10 text-slate-100 text-sm rounded-2xl rounded-bl-sm px-4 py-2.5';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
}

function showTypingIndicator() {
  const log = document.getElementById('assistant-log');
  if (!log) return;
  const wrap = document.createElement('div');
  wrap.id = 'typing-indicator';
  wrap.className = 'flex justify-start';
  wrap.innerHTML = `<div class="bg-white/5 border border-white/10 rounded-2xl rounded-bl-sm px-4 py-3 flex gap-1">
    <span class="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style="animation-delay:0ms"></span>
    <span class="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style="animation-delay:150ms"></span>
    <span class="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style="animation-delay:300ms"></span>
  </div>`;
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function speakText(text) {
  if (!('speechSynthesis' in window)) return;
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.02;
  utter.pitch = 1.1;
  const voices = window.speechSynthesis.getVoices();
  const femaleVoice = voices.find(v => /female|samantha|victoria|zira|google us english/i.test(v.name));
  if (femaleVoice) utter.voice = femaleVoice;
  window.speechSynthesis.speak(utter);
}

async function sendAssistantMessage(text) {
  if (!text || !text.trim()) return;
  appendAssistantMessage(text, 'user');
  const input = document.getElementById('assistant-input');
  if (input) input.value = '';
  showTypingIndicator();

  try {
    const res = await fetch('/api/assistant/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    setTimeout(() => {
      removeTypingIndicator();
      appendAssistantMessage(data.reply, 'bot');
      if (window.__voiceEnabled) speakText(data.reply);
    }, 500);
  } catch (err) {
    removeTypingIndicator();
    appendAssistantMessage("I'm having trouble connecting right now. Please try again.", 'bot');
  }
}

function handleAssistantSubmit(e) {
  e.preventDefault();
  const input = document.getElementById('assistant-input');
  sendAssistantMessage(input.value);
}

function askSuggested(text) {
  sendAssistantMessage(text);
}

function toggleAssistantVoice() {
  window.__voiceEnabled = !window.__voiceEnabled;
  const btn = document.getElementById('voice-toggle-btn');
  if (btn) btn.classList.toggle('text-primary', window.__voiceEnabled);
}

// Voice input via Web Speech API (if supported)
function startVoiceInput() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    appendAssistantMessage("Voice input isn't supported in this browser.", 'bot');
    return;
  }
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    sendAssistantMessage(transcript);
  };
  recognition.start();
}
