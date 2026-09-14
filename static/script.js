// JavaScript Application Logic - Student Performance Dashboard
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadDataset();
    initPredictor();
});

// 1. Tab Switching Logic
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });
}

// 2. Lightbox Modal Logic
function openLightbox(imgSrc, title) {
    const modal = document.getElementById('lightbox-modal');
    const modalImg = document.getElementById('lightbox-img');
    const modalTitle = document.getElementById('lightbox-title');

    modalImg.src = imgSrc;
    modalTitle.textContent = title;
    modal.style.display = 'flex';
}

function closeLightbox(event) {
    const modal = document.getElementById('lightbox-modal');
    if (!event || event.target === modal || event.target.classList.contains('lightbox-close')) {
        modal.style.display = 'none';
    }
}

// 3. Load & Render Cleaned Dataset
let datasetRecords = [];

async function loadDataset() {
    try {
        const response = await fetch('/api/dataset');
        datasetRecords = await response.json();
        renderTable(datasetRecords);
    } catch (err) {
        console.error('Failed to load dataset:', err);
    }
}

function renderTable(records) {
    const tbody = document.getElementById('table-body');
    if (!records || records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="12" class="text-center">No student records found.</td></tr>';
        return;
    }

    tbody.innerHTML = records.map((r, i) => {
        const perfClass = r.performance_level === 'High' ? 'tag-emerald' : 
                          r.performance_level === 'Medium' ? 'tag-blue' : 'tag-gold';
        return `
            <tr>
                <td>${i + 1}</td>
                <td>${r.gender}</td>
                <td>${r['race/ethnicity']}</td>
                <td>${r['parental level of education']}</td>
                <td>${r.lunch}</td>
                <td>${r['test preparation course']}</td>
                <td><strong>${r['math score']}</strong></td>
                <td><strong>${r['reading score']}</strong></td>
                <td><strong>${r['writing score']}</strong></td>
                <td><span class="tag tag-purple">${r.average_score}</span></td>
                <td><span class="tag tag-pink">${r.grade}</span></td>
                <td><span class="tag ${perfClass}">${r.performance_level}</span></td>
            </tr>
        `;
    }).join('');
}

// Search Filter Logic
document.getElementById('data-search')?.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    const filtered = datasetRecords.filter(r => {
        return Object.values(r).some(val => String(val).toLowerCase().includes(query));
    });
    renderTable(filtered);
});

// 4. Live Predictor API Logic
function initPredictor() {
    const form = document.getElementById('predict-form');
    const resultBox = document.getElementById('result-box');

    form?.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
            gender: document.getElementById('input-gender').value,
            race: document.getElementById('input-race').value,
            education: document.getElementById('input-edu').value,
            lunch: document.getElementById('input-lunch').value,
            prep: document.getElementById('input-prep').value
        };

        resultBox.innerHTML = `
            <div class="result-placeholder">
                <i class="fa-solid fa-spinner fa-spin"></i>
                <h3>Running Decision Tree Classifier...</h3>
            </div>
        `;

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (data.success) {
                renderPredictionResult(data, payload);
            } else {
                resultBox.innerHTML = `<div class="result-placeholder"><p class="text-rose">Error: ${data.error}</p></div>`;
            }
        } catch (err) {
            resultBox.innerHTML = `<div class="result-placeholder"><p class="text-rose">Prediction failed. Server connection error.</p></div>`;
        }
    });
}

function renderPredictionResult(data, input) {
    const resultBox = document.getElementById('result-box');
    const perf = data.predicted_performance;
    const status = data.predicted_status;
    const passProb = data.pass_probability;
    const probs = data.performance_probabilities;

    const perfColor = perf === 'High' ? 'var(--emerald)' : perf === 'Medium' ? 'var(--blue)' : 'var(--amber)';
    const statusBadge = status === 'Pass' ? 'badge-success' : 'badge-gray';

    resultBox.innerHTML = `
        <div class="result-card-active">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                <div>
                    <span class="result-title-badge" style="background: rgba(99,102,241,0.15); color: ${perfColor}; border: 1px solid ${perfColor};">
                        <i class="fa-solid fa-brain"></i> Performance Level: ${perf}
                    </span>
                    <h3 style="font-family: var(--font-heading); font-size: 1.6rem; font-weight: 800;">
                        Overall Status: <span style="color: ${status === 'Pass' ? 'var(--emerald)' : 'var(--rose)'};">${status}</span>
                    </h3>
                </div>
                <span class="badge ${statusBadge}" style="font-size: 0.9rem;">
                    Pass Probability: ${passProb}%
                </span>
            </div>

            <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1.2rem;">
                Prediction generated for <strong>${input.gender}</strong> | <strong>${input.education}</strong> | <strong>${input.lunch} lunch</strong> | <strong>Prep: ${input.prep}</strong>
            </p>

            <h4 style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.8rem; color: var(--text-muted);">Class Probability Breakdown:</h4>

            <div class="prob-bar-container">
                <div class="prob-label"><span>High Performance Level</span><span>${probs['High'] || 0}%</span></div>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: ${probs['High'] || 0}%; background: var(--emerald);"></div></div>
            </div>

            <div class="prob-bar-container">
                <div class="prob-label"><span>Medium Performance Level</span><span>${probs['Medium'] || 0}%</span></div>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: ${probs['Medium'] || 0}%; background: var(--blue);"></div></div>
            </div>

            <div class="prob-bar-container">
                <div class="prob-label"><span>Low Performance Level</span><span>${probs['Low'] || 0}%</span></div>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: ${probs['Low'] || 0}%; background: var(--amber);"></div></div>
            </div>
        </div>
    `;
}
