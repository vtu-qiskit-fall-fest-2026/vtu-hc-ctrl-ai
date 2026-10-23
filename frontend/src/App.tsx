import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { MicrogridSample, ModelMetrics, LowDataBenchmarkPoint } from './types/microgrid';
import { generateSyntheticMicrogridData } from './services/dataGenerator';
import { preparePreprocessedDataset, DatasetSplit } from './services/preprocessing';
import { computeQuantumKernelMatrix } from './services/quantumEngine';
import { trainClassicalSVC, trainQuantumQSVC, predictBatch, SVCTrainedModel } from './services/svmSolver';
import { calculateMetrics, runLowDataExperiment } from './services/evaluation';
import { Header, PageId } from './components/Header';
import Sidebar from './components/Sidebar';
import { OverviewPage } from './components/OverviewPage';
import { LiveSimulationPage } from './components/LiveSimulationPage';
import { ScenarioLabPage } from './components/ScenarioLabPage';
import { AnomalyInvestigationPage } from './components/AnomalyInvestigationPage';
import { QuantumLabPage } from './components/QuantumLabPage';
import { BenchmarkPage } from './components/BenchmarkPage';

import {
  ChevronUp,
  ChevronDown,
  ArrowUp,
  Sparkles,
  Layers,
  Activity,
  Zap,
  Eye,
  Radio,
  Atom,
  BarChart3,
} from 'lucide-react';

export default function App() {
  const [currentPage, setCurrentPage] = useState<PageId>('overview');
  const [randomSeed, setRandomSeed] = useState<number>(42);
  const [sampleCount] = useState<number>(100);
  const [samples, setSamples] = useState<MicrogridSample[]>([]);
  const [theme, setTheme] = useState<'dark' | 'light'>('light');

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  const [isTraining, setIsTraining] = useState<boolean>(true);
  const [classicalModel, setClassicalModel] = useState<SVCTrainedModel | null>(null);
  const [quantumModel, setQuantumModel] = useState<SVCTrainedModel | null>(null);
  const [classicalMetrics, setClassicalMetrics] = useState<ModelMetrics | null>(null);
  const [quantumMetrics, setQuantumMetrics] = useState<ModelMetrics | null>(null);
  const [kernelMatrix, setKernelMatrix] = useState<number[][] | null>(null);
  const [lowDataPoints, setLowDataPoints] = useState<LowDataBenchmarkPoint[]>([]);
  const [datasetSplit, setDatasetSplit] = useState<DatasetSplit | null>(null);

  const trainAndBenchmark = useCallback((rawSamples: MicrogridSample[], seed: number) => {
    setIsTraining(true);

    setTimeout(() => {
      try {
        const split = preparePreprocessedDataset(
          rawSamples,
          ['demand_kw', 'solar_kw', 'voltage_dev_v', 'frequency_dev_hz'],
          0.25,
          seed
        );
        setDatasetSplit(split);

        const cModel = trainClassicalSVC(split.X_train, split.y_train, 1.0, 0.5);
        const cBatch = predictBatch(cModel, split.X_test);
        const cMetrics = calculateMetrics(
          cBatch.predictions,
          split.y_test,
          cModel.trainTimeMs,
          cBatch.inferenceTimeMs
        );
        setClassicalModel(cModel);
        setClassicalMetrics(cMetrics);

        const numQubits = 4;
        const reps = 2;
        const tK0 = performance.now();
        const K_train = computeQuantumKernelMatrix(split.X_train, split.X_train, numQubits, reps);
        const tK1 = performance.now();
        const kComputeTimeMs = Math.round(tK1 - tK0);
        setKernelMatrix(K_train);

        const qModel = trainQuantumQSVC(
          split.X_train,
          split.y_train,
          K_train,
          numQubits,
          reps,
          1.0,
          kComputeTimeMs
        );
        const qBatch = predictBatch(qModel, split.X_test);
        const qMetrics = calculateMetrics(
          qBatch.predictions,
          split.y_test,
          qModel.trainTimeMs,
          qBatch.inferenceTimeMs
        );
        setQuantumModel(qModel);
        setQuantumMetrics(qMetrics);

        const lowData = runLowDataExperiment(
          split.X_train,
          split.y_train,
          split.X_test,
          split.y_test,
          numQubits,
          reps
        );
        setLowDataPoints(lowData);
      } catch (err) {
        console.error('Error in model training:', err);
      } finally {
        setIsTraining(false);
      }
    }, 50);
  }, []);

  useEffect(() => {
    const initialSamples = generateSyntheticMicrogridData(sampleCount, randomSeed);
    setSamples(initialSamples);
    trainAndBenchmark(initialSamples, randomSeed);
  }, []);

  const handleResetData = () => {
    const newSeed = Math.floor(Math.random() * 10000);
    setRandomSeed(newSeed);
    const newSamples = generateSyntheticMicrogridData(sampleCount, newSeed);
    setSamples(newSamples);
    trainAndBenchmark(newSamples, newSeed);
  };

  const handleUploadCustomData = (newSamples: MicrogridSample[]) => {
    setSamples(newSamples);
    trainAndBenchmark(newSamples, randomSeed);
  };

  const handleRerunBenchmark = () => {
    trainAndBenchmark(samples, randomSeed);
  };

  const tabSequence = useMemo(
    () => [
      { id: 'overview' as PageId, label: 'Overview', number: '01', subtitle: 'Executive Dashboard & Real-Time Telemetry', accent: 'cyan' },
      { id: 'scenario_lab' as PageId, label: 'Scenario Lab', number: '02', subtitle: 'Physics Compliance & Transient Injector', accent: 'amber' },
      { id: 'investigation' as PageId, label: 'Investigation', number: '03', subtitle: 'State Space & Anomaly Forensics', accent: 'cyan' },
      { id: 'live_stream' as PageId, label: 'Live Stream', number: '04', subtitle: 'Continuous Telemetry & Dual ML Stream', accent: 'rose' },
      { id: 'quantum_lab' as PageId, label: 'Quantum Circuit', number: '05', subtitle: 'Qiskit ZZFeatureMap Hilbert Space', accent: 'violet' },
      { id: 'benchmark' as PageId, label: 'Benchmark', number: '06', subtitle: 'Empirical Scikit-Learn vs QSVC Comparison', accent: 'sky' },
    ],
    []
  );

  const [scrollPercent, setScrollPercent] = useState<number>(0);
  const isManualScrolling = useRef<boolean>(false);

  const handleSelectPage = useCallback((pageId: PageId) => {
    setCurrentPage(pageId);
    isManualScrolling.current = true;
    const el = document.getElementById(`section-${pageId}`);
    if (el) {
      const headerOffset = 80;
      const elementPosition = el.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
      window.scrollTo({ top: offsetPosition, behavior: 'smooth' });
      el.classList.remove('section-targeted');
      void el.offsetWidth;
      el.classList.add('section-targeted');
      setTimeout(() => {
        el.classList.remove('section-targeted');
      }, 1300);
    }
    setTimeout(() => {
      isManualScrolling.current = false;
    }, 850);
  }, []);

  const isDark = theme === 'dark';

  return (
    <div className={`min-h-screen flex flex-col font-sans transition-colors duration-200 ${isDark ? 'bg-[#090d16] text-slate-100' : 'bg-white text-slate-900'}`}>
      <div className="xl:pl-72 transition-all">
        <Header currentPage={currentPage} onSelectPage={handleSelectPage} datasetCount={samples.length} isTraining={isTraining} onResetData={handleResetData} theme={theme} onToggleTheme={toggleTheme} />
      </div>

      <Sidebar activeSection={currentPage} onSelectSection={(id) => {
        const pageIdMap: Record<string, PageId> = {
          'overview': 'overview',
          'scenario-lab': 'scenario_lab',
          'scenario_lab': 'scenario_lab',
          'investigation': 'investigation',
          'live-stream': 'live_stream',
          'live_stream': 'live_stream',
          'quantum-circuit': 'quantum_lab',
          'quantum_lab': 'quantum_lab',
          'benchmark': 'benchmark',
        };
        const target = pageIdMap[id] || (id as PageId);
        handleSelectPage(target);
      }} className="hidden xl:flex" />

      <main className="flex-1 space-y-20 pb-28 xl:pl-[220px]">
        <div id="overview" className="relative -top-24 pointer-events-none" aria-hidden="true" />
        <section id="section-overview" data-tab="overview" className={`tab-section ${currentPage === 'overview' ? 'section-active' : ''}`}>
          <OverviewPage samples={samples} classicalModel={classicalModel} quantumModel={quantumModel} classicalMetrics={classicalMetrics} quantumMetrics={quantumMetrics} scaler={datasetSplit?.scaler || null} onNavigateToScenarioLab={() => handleSelectPage('scenario_lab')} onNavigateToQuantumLab={() => handleSelectPage('quantum_lab')} theme={theme} />
        </section>
      </main>
    </div>
  );
}
