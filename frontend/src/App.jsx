import { useState, useEffect } from 'react';
import './index.css';

const API_BASE = 'http://localhost:8000/api';

function App() {
  // Pipeline State
  const [kaggleFileName, setKaggleFileName] = useState('AyurGenixAI_Dataset.csv');
  const [kaggleFilePath, setKaggleFilePath] = useState('kagglekirti123/ayurgenixai-ayurvedic-dataset');
  
  const [promptMapping, setPromptMapping] = useState([
    { key: 'Symptoms', value: 'Symptoms' },
    { key: 'Severity', value: 'Symptom Severity' },
    { key: 'Patient Profile', value: '{Age Group} | {Gender}' },
    { key: 'Medical History', value: 'Medical History' },
    { key: 'Current Medications', value: 'Current Medications' },
    { key: 'Risk Factors', value: 'Risk Factors' },
    { key: 'Environmental Factors', value: 'Environmental Factors' },
    { key: 'Lifestyle & Diet', value: '{Occupation and Lifestyle} | Diet: {Dietary Habits}' },
    { key: 'Stress & Sleep', value: 'Stress - {Stress Levels}, Sleep - {Sleep Patterns}' }
  ]);

  const [completionMapping, setCompletionMapping] = useState([
    { key: 'Diagnosis', value: '{Disease} (Hindi: {Hindi Name}, Marathi: {Marathi Name})' },
    { key: 'Dosha Imbalance', value: 'Doshas' },
    { key: 'Prakriti (Constitution)', value: 'Constitution/Prakriti' },
    { key: '', value: '\\n--- Ayurvedic Treatment Plan ---' },
    { key: 'Herbs', value: 'Ayurvedic Herbs' },
    { key: 'Formulation', value: 'Formulation' },
    { key: 'Diet & Lifestyle Recommendations', value: 'Diet and Lifestyle Recommendations' },
    { key: 'Yoga & Physical Therapy', value: 'Yoga & Physical Therapy' },
    { key: ' ', value: '\\n--- Additional Guidance ---' },
    { key: 'General Recommendations', value: 'Patient Recommendations' },
    { key: 'Prevention', value: 'Prevention' },
    { key: 'Prognosis', value: 'Prognosis' }
  ]);

  const [promptHeader, setPromptHeader] = useState('Based on the following patient profile, symptoms, and medical history, provide an accurate diagnosis and a comprehensive Ayurvedic treatment plan.\\n\\n--- Patient Details ---');
  const [systemPrompt, setSystemPrompt] = useState('You are an expert Ayurvedic medical assistant. You provide accurate diagnoses and comprehensive Ayurvedic treatment plans based on patient symptoms, profiles, and medical history.');

  const [pipelineJobId, setPipelineJobId] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(null);

  // Finetune State
  const [finetuneJobId, setFinetuneJobId] = useState(null);
  const [finetuneStatus, setFinetuneStatus] = useState(null);

  // Mappings Handlers
  const handleMappingChange = (setter, index, field, newValue) => {
    setter(prev => {
      const updated = [...prev];
      updated[index][field] = newValue;
      return updated;
    });
  };

  const addMappingRow = (setter) => setter(prev => [...prev, { key: '', value: '' }]);
  const removeMappingRow = (setter, index) => setter(prev => prev.filter((_, i) => i !== index));

  // Convert array back to dict for API
  const getMappingDict = (arr) => {
    const dict = {};
    arr.forEach(item => {
      dict[item.key] = item.value;
    });
    return dict;
  };

  const startPipeline = async () => {
    try {
      const res = await fetch(`${API_BASE}/pipeline/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          kaggle_file_name: kaggleFileName,
          kaggle_file_path: kaggleFilePath,
          prompt_mapping: getMappingDict(promptMapping),
          completion_mapping: getMappingDict(completionMapping),
          prompt_header: promptHeader,
          system_prompt: systemPrompt
        })
      });
      const data = await res.json();
      setPipelineJobId(data.job_id);
    } catch (err) {
      console.error(err);
      alert('Failed to start pipeline');
    }
  };

  const startFinetuning = async () => {
    try {
      // Extract base name logic roughly
      const baseName = kaggleFileName.split('.')[0];
      const res = await fetch(`${API_BASE}/pipeline/finetune`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: `${baseName}-dataset`,
          model_name: `${baseName}-model`
        })
      });
      const data = await res.json();
      setFinetuneJobId(data.job_id);
    } catch (err) {
      console.error(err);
      alert('Failed to start finetuning');
    }
  };

  // Polling Effects
  useEffect(() => {
    if (!pipelineJobId) return;
    
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/pipeline/status/${pipelineJobId}`);
        const data = await res.json();
        setPipelineStatus(data);
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error(err);
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, [pipelineJobId]);

  useEffect(() => {
    if (!finetuneJobId) return;
    
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/pipeline/status/${finetuneJobId}`);
        const data = await res.json();
        setFinetuneStatus(data);
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error(err);
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [finetuneJobId]);


  return (
    <div className="app-container">
      <header className="header">
        <h1>Universal Finetuning Hub</h1>
        <p>End-to-End orchestration from raw Kaggle Datasets to custom Pioneer Models</p>
      </header>

      <div className="glass-panel">
        <h2>1. Job Configuration</h2>
        
        <div className="form-group">
          <label>Kaggle File Name <span title="The filename of the dataset as it appears on Kaggle (e.g. AyurGenixAI_Dataset.csv)" className="tooltip">❓</span></label>
          <input 
            type="text" 
            value={kaggleFileName} 
            onChange={e => setKaggleFileName(e.target.value)} 
          />
        </div>

        <div className="form-group">
          <label>Kaggle Dataset Path <span title="The full repository path on Kaggle (e.g. kagglekirti123/ayurgenixai-ayurvedic-dataset)" className="tooltip">❓</span></label>
          <input 
            type="text" 
            value={kaggleFilePath} 
            onChange={e => setKaggleFilePath(e.target.value)} 
          />
        </div>

        <div className="form-group" style={{marginTop: '1.5rem'}}>
          <label>Prompt Header <span title="The instruction text that precedes the mapped prompt fields in your final JSONL. Sets the stage for the task." className="tooltip">❓</span></label>
          <textarea 
            className="mapping-input" 
            style={{minHeight: '80px', resize: 'vertical'}}
            placeholder="e.g. Based on the following details, provide..."
            value={promptHeader} 
            onChange={e => setPromptHeader(e.target.value)} 
          />
        </div>

        <div className="form-group">
          <label>System Prompt (for SFT) <span title="The overarching system message injected into the final conversation format for model finetuning." className="tooltip">❓</span></label>
          <textarea 
            className="mapping-input" 
            style={{minHeight: '80px', resize: 'vertical'}}
            placeholder="e.g. You are an expert assistant..."
            value={systemPrompt} 
            onChange={e => setSystemPrompt(e.target.value)} 
          />
        </div>

        <div className="grid-2">
          {/* Prompt Mappings */}
          <div>
            <label className="section-label">Prompt Mapping <span title="Define which CSV columns map to the input 'prompt' features. Values can use {ColumnName} template syntax." className="tooltip">❓</span></label>
            {promptMapping.map((item, idx) => (
              <div className="mapping-row" key={`p-${idx}`}>
                <input 
                  type="text" 
                  className="mapping-input"
                  placeholder="Key (e.g. Symptoms)" 
                  value={item.key} 
                  onChange={e => handleMappingChange(setPromptMapping, idx, 'key', e.target.value)}
                />
                <input 
                  type="text" 
                  className="mapping-input"
                  placeholder="Value/Format" 
                  value={item.value} 
                  onChange={e => handleMappingChange(setPromptMapping, idx, 'value', e.target.value)}
                />
                <button className="btn-remove" onClick={() => removeMappingRow(setPromptMapping, idx)}>✕</button>
              </div>
            ))}
            <button className="btn-add" onClick={() => addMappingRow(setPromptMapping)}>+ Add Prompt Field</button>
          </div>

          {/* Completion Mappings */}
          <div>
            <label className="section-label">Completion Mapping <span title="Define which CSV columns map to the expected output 'completion'. Values can use {ColumnName} template syntax." className="tooltip">❓</span></label>
            {completionMapping.map((item, idx) => (
              <div className="mapping-row" key={`c-${idx}`}>
                <input 
                  type="text" 
                  className="mapping-input"
                  placeholder="Key (e.g. Diagnosis)" 
                  value={item.key} 
                  onChange={e => handleMappingChange(setCompletionMapping, idx, 'key', e.target.value)}
                />
                <input 
                  type="text" 
                  className="mapping-input"
                  placeholder="Value/Format" 
                  value={item.value} 
                  onChange={e => handleMappingChange(setCompletionMapping, idx, 'value', e.target.value)}
                />
                <button className="btn-remove" onClick={() => removeMappingRow(setCompletionMapping, idx)}>✕</button>
              </div>
            ))}
            <button className="btn-add" onClick={() => addMappingRow(setCompletionMapping)}>+ Add Completion Field</button>
          </div>
        </div>

        <button 
          className="btn-primary" 
          onClick={startPipeline}
          disabled={pipelineJobId && (!pipelineStatus || pipelineStatus?.status === 'running')}
        >
          {pipelineJobId ? 'Pipeline Started' : 'Start Pipeline orchestration'}
        </button>

        {pipelineStatus && (
          <div className="status-indicator">
            {pipelineStatus.status === 'running' && <div className="spinner"></div>}
            <div className="status-text">{pipelineStatus.message}</div>
            <div className={`status-badge ${pipelineStatus.status}`}>{pipelineStatus.status}</div>
          </div>
        )}
      </div>

      {pipelineStatus?.status === 'completed' && pipelineStatus.files && (
        <div className="glass-panel">
          <h2>2. Processed Artifacts</h2>
          <p style={{color: 'var(--color-text-secondary)', marginBottom: '1rem', fontSize: '1.125rem'}}>Download the generated files from the pipeline steps.</p>
          
          <div className="download-grid">
            <a href={`${API_BASE}/download?path=${encodeURIComponent(pipelineStatus.files.raw_jsonl)}`} className="download-card" target="_blank" rel="noreferrer">
              <span style={{fontSize: '2rem'}}>📄</span>
              <strong>Raw JSONL</strong>
            </a>
            <a href={`${API_BASE}/download?path=${encodeURIComponent(pipelineStatus.files.processed_file)}`} className="download-card" target="_blank" rel="noreferrer">
              <span style={{fontSize: '2rem'}}>⚙️</span>
              <strong>Adaption Processed</strong>
            </a>
            <a href={`${API_BASE}/download?path=${encodeURIComponent(pipelineStatus.files.sft_jsonl)}`} className="download-card" target="_blank" rel="noreferrer">
              <span style={{fontSize: '2rem'}}>🎯</span>
              <strong>Converted SFT JSONL</strong>
            </a>
          </div>
        </div>
      )}

      {pipelineStatus?.status === 'completed' && (
        <div className="glass-panel">
          <h2>3. Pioneer Finetuning</h2>
          <p style={{color: 'var(--color-text-secondary)', marginBottom: '1.5rem', fontSize: '1.125rem'}}>Push the resulting dataset to Pioneer for LoRA fine-tuning.</p>
          
          <button 
            className="btn-primary" 
            onClick={startFinetuning}
            disabled={finetuneJobId && (!finetuneStatus || finetuneStatus?.status === 'running')}
            style={{backgroundColor: 'var(--color-success)'}}
          >
            {finetuneJobId ? 'Finetuning Started' : '🚀 Start Pioneer Finetuning'}
          </button>

          {finetuneStatus && (
            <div className="status-indicator">
              {finetuneStatus.status === 'running' && <div className="spinner" style={{borderTopColor: '#10b981'}}></div>}
              <div className="status-text">{finetuneStatus.message}</div>
              <div className={`status-badge ${finetuneStatus.status}`}>{finetuneStatus.status}</div>
            </div>
          )}

          {finetuneStatus?.status === 'completed' && (
            <div className="inference-box">
              <h3>Model Training Complete! 🎉</h3>
              <p>Your custom model is ready for inference. Endpoint ID:</p>
              <code>{finetuneStatus.inference_endpoint}</code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
