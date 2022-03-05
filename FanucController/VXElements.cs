using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Diagnostics;
using VXelementsApi;
using VXelementsApi.Tracker;
using VXelementsApi.VXtrack;
using VXelementsApi.Types;
using MathNet.Numerics.LinearAlgebra;
using Serilog;

namespace FanucController
{
    public class VXelementsUtility
    {

        #region Fields

        // Api interfaces
        private IVXelements iVXelements;
        private ITracker iTracker;
        private IVXtrack iVXtrack;
        private bool trackerAttached;
        private bool vxtrackAttached;

        // Invokers
        private VXeventInvoker modelDetectionStartedInvoker;
        private DetectModelStoppedEventInvoker modelDetectionStoppedInvoker;
        private VXeventInvoker modelChangedInvoker;
        private VXeventInvoker trackingDataReadyInvoker;
        private VXeventInvoker trackingStartedInvoker;
        private VXeventInvoker trackingStoppedInvoker;
        private VXeventInvoker trackerPoseChangedInvoker;
        private VXeventInvoker sequenceChangedInvoker;

        // Tracking interfaces
        private ITrackingEntity iTrackingEntity;
        private ITrackingSequence iTrackingSequence;
        private ITrackingParameters iTrackingParameters;
        private ITrackingModelRelation iTrackingModelRelation;
        private ITrackingModel iTrackingModel;
        private List<ITrackingModel> iTrackingModelList;
        private List<ITrackingSequence> iTrackingSequenceList;

        // Memory & IO
        private Buffer<PoseData>[] poseBuffers;
        private readonly object poseLock = new object();

        // Pose data
        private IPose3d poseRaw;
        private Vector<double> poseTemp;
        private Vector<double>[] poseCameraFrame;
        public Vector<double> PoseCameraFrame
        {
            get
            {
                lock (poseLock) { return poseCameraFrame[0]; }
            }
            set
            {
                lock (poseLock) { poseCameraFrame[0] = value; }
            }
        }
        public Vector<double> PoseCameraFrameKf
        {
            get
            {
                lock (poseLock) { return poseCameraFrame[1]; }
            }
            set
            {
                lock (poseLock) { poseCameraFrame[1] = value; }
            }
        }
        public Vector<double> PoseCameraFrameRkf
        {
            get
            {
                lock (poseLock) { return poseCameraFrame[2]; }
            }
            set
            {
                lock (poseLock) { poseCameraFrame[2] = value; }
            }
        }
        public bool[] filterActivated;

        // Stopwatch
        private Stopwatch stopWatch;
        private double time;

        // Thread 
        private Thread eventThread;

        // Kalman Filter
        private KalmanFilter standardKalman;
        private RobustKalmanFilter robustKalman;
        private bool standardKalmanEnabled;
        private bool robustKalmanEnabled;

        #endregion

        #region Constructor

        public VXelementsUtility(Stopwatch stopWatch)
        {
            // Api
            trackerAttached = false;
            vxtrackAttached = false;

            // Invoker
            modelDetectionStartedInvoker = new VXeventInvoker(new VXeventHandler(ModelDetectionStarted));
            modelDetectionStoppedInvoker = new DetectModelStoppedEventInvoker(new DetectModelStoppedEventHandler(ModelDetectionStopped));
            modelChangedInvoker = new VXeventInvoker(new VXeventHandler(ModelChanged));
            trackingDataReadyInvoker = new VXeventInvoker(new VXeventHandler(TrackingDataReady));
            trackingStartedInvoker = new VXeventInvoker(new VXeventHandler(TrackingStarted));
            trackingStoppedInvoker = new VXeventInvoker(new VXeventHandler(TrackingStopped));
            trackerPoseChangedInvoker = new VXeventInvoker(new VXeventHandler(TrackerPoseChanged));
            
            // Thread
            eventThread = new Thread(new ThreadStart(ThreadDataReady));
            eventThread.IsBackground = true;
            
            // Tracking model list
            iTrackingModelList = new List<ITrackingModel>();
            
            // Tracking sequence list
            iTrackingSequenceList = new List<ITrackingSequence>();
            
            // Kalman filter
            standardKalman = new KalmanFilter();
            standardKalmanEnabled = false;
            robustKalman = new RobustKalmanFilter();
            robustKalmanEnabled = false;
            poseTemp = CreateVector.Dense<double>(6);
            poseCameraFrame = new Vector<double>[3];
            poseCameraFrame[0] = CreateVector.Dense<double>(6); // raw
            poseCameraFrame[1] = CreateVector.Dense<double>(6); // kf
            poseCameraFrame[2] = CreateVector.Dense<double>(6); // rkf
            filterActivated = new bool[3];

            // Buffer
            int buffer_size = 20000;
            poseBuffers = new Buffer<PoseData>[3];
            poseBuffers[0] = new Buffer<PoseData>(buffer_size); // raw
            poseBuffers[1] = new Buffer<PoseData>(buffer_size); // kf
            poseBuffers[2] = new Buffer<PoseData>(buffer_size); // rkf

            // Stopwatch
            this.stopWatch = stopWatch;
            time = 0;
        }

        #endregion

        #region Actions

        public void TestAction()
        {
            ConnectApi();
            AttachToVXelementsEvents();
            Log.Information("Connected!");
        }

        public void QuickConnect(string targetsPath, string modelPath)
        {
            ConnectApi();
            AttachToVXelementsEvents();
            OpenPositioningTargets(targetsPath);
            ImportModel(modelPath);
            //CreateSequence(iTrackingModelList.ToArray());
        }

        public void Reset(bool hard = false)
        {
            robustKalmanEnabled = false;
            ResetVXTrack();
            iTrackingModelList.Clear();
            iTrackingSequenceList.Clear();
            if (hard)
            {
                DetachFromVXelementsEvents();
                ResetApi();
            }
        }

        public void Exit()
        {
            ExitApi();
        }

        #endregion

        #region Basic Functions

        public void ConnectApi()
        {
            try
            {
                if (!ApiManager.IsConnected)
                {
                    ApiManager.Connect();
                    ApiManager.Disconnected += ApiDisconnected;
                }
                if (iVXelements == null)
                {
                    iVXelements = ApiManager.VXelements;
                    iTracker = iVXelements.Tracker;
                    iVXtrack = iVXelements.VXtrack;
                }

            }
            catch (VXelementsException ex)
            {
                Console.WriteLine(ex.ToString());
                ConnectApi();
            }
            Log.Information("Connected Api");
        }

        public void ResetApi()
        {
            iVXelements = null;
            iTracker = null;
            iVXtrack = null;
            vxtrackAttached = false;
            trackerAttached = false;
        }

        public void ExitApi()
        {
            try
            {
                if (iVXelements != null)
                {
                    DetachFromVXelementsEvents();
                    iVXelements.Exit();
                }
            }
            catch { }
            Log.Information("Exited Api");
        }

        public void AttachToVXelementsEvents()
        {
            try
            {
                // Add invoker (event handlers) to event
                if (!trackerAttached)
                {
                    iTracker.TrackerPoseChangedEvent.AddEvent(trackerPoseChangedInvoker);
                    trackerAttached = true;
                }
                if (!vxtrackAttached)
                {
                    iVXtrack.DetectModelStartedEvent.AddEvent(modelDetectionStartedInvoker);
                    iVXtrack.ModelsChangedEvent.AddEvent(modelChangedInvoker);
                    iVXtrack.TrackingDataReadyEvent.AddEvent(trackingDataReadyInvoker);
                    iVXtrack.TrackingStartedEvent.AddEvent(trackingStartedInvoker);
                    iVXtrack.TrackingStoppedEvent.AddEvent(trackingStoppedInvoker);
                    iVXtrack.SequencesChangedEvent.AddEvent(sequenceChangedInvoker);
                    vxtrackAttached = true;
                }

            }
            catch
            {
                ExitApi();
                ResetApi();
                System.Threading.Thread.Sleep(1000);
                ConnectApi();
                AttachToVXelementsEvents();
            }
            Log.Information("Attached Vx events");
        }

        public void DetachFromVXelementsEvents()
        {
            try
            {
                // Remove invoker (event handlers) to event
                if (trackerAttached)
                {
                    iTracker.TrackerPoseChangedEvent.RemoveEvent(trackerPoseChangedInvoker);
                    trackerAttached = false;
                }
                if (vxtrackAttached)
                {
                    iVXtrack.DetectModelStartedEvent.RemoveEvent(modelDetectionStartedInvoker);
                    iVXtrack.ModelsChangedEvent.RemoveEvent(modelChangedInvoker);
                    iVXtrack.TrackingDataReadyEvent.RemoveEvent(trackingDataReadyInvoker);
                    iVXtrack.TrackingStartedEvent.RemoveEvent(trackingStartedInvoker);
                    iVXtrack.TrackingStoppedEvent.RemoveEvent(trackingStoppedInvoker);
                    iVXtrack.SequencesChangedEvent.RemoveEvent(sequenceChangedInvoker);
                    vxtrackAttached = false;
                }
            }
            catch
            {
                ExitApi();
            }
            Log.Information("Detached Vx events");
        }

        public void ClearSequences()
        {
            iVXtrack.RemoveAllSequences();
        }

        public void ClearEntities()
        {
            iVXtrack.RemoveAllEntities();
        }

        public void ResetVXTrack()
        {
            ClearEntities();
            ClearSequences();
        }

        public void OpenPositioningTargets(string targetPath = null)
        {
            iVXelements.OpenPositioningTargets(targetPath);
        }

        public void ImportModel(string modelPath = null)
        {
            iTrackingModelList.Add(iVXtrack.ImportModel(modelPath));
        }

        public void SaveModel(string savePath, ITrackingModel model)
        {
            model.Save(savePath);
        }

        public void DetectModel()
        {
            iVXtrack.DetectModel();
        }

        public void CreateSequence(ITrackingEntity[] trackingEntity = null)
        {
            iTrackingSequenceList.Add(iVXtrack.CreateSequence(trackingEntity));
        }

        public void ExportSequence(ITrackingSequence trackingSequence, string exportPath = null)
        {
            trackingSequence.ExportToCsv(exportPath);
        }

        public void AddModelRelation(ITrackingSequence currentSequence, ITrackingModel observedModel, ITrackingModel referenceModel)
        {
            currentSequence.AddTrackingModelRelation(observedModel, referenceModel);
        }

        public void StartTracking(ITrackingSequence trackingSequence = null)
        {
            if (trackingSequence == null) 
            {
                iVXtrack.StartTracking();
            }
            else
            {
                iVXtrack.StartTracking(trackingSequence);
            }
        }

        public void StopTracking()
        {
            iVXtrack.StopTracking();
        }

        public IPose3d GetLastPose(ITrackingEntity trackingEntity)
        {
            var currentSequence = iVXtrack.CurrentTrackingSequence;
            return currentSequence.GetLastPose(trackingEntity);
        }

        public void ShowGraphics()
        {
            iVXtrack.ShowGraphicsViewForm();
            iVXtrack.ShowTableViewForm();
            iVXtrack.ShowProjectionViewForm();
        }

        public void HideGraphics()
        {
            iVXtrack.HideGraphicsViewForm();
            iVXtrack.HideTableViewForm();
            iVXtrack.HideProjectionViewForm();
        }

        #endregion

        #region Data Processing

        private void ProcessPose3d(ITrackingEntity trackingEntity)
        {
            poseRaw = GetLastPose(trackingEntity);
            time = stopWatch.Elapsed.TotalSeconds;
            if (poseRaw.Valid)
            {
                poseTemp[0] = poseRaw.Translation.X;
                poseTemp[1] = poseRaw.Translation.Y;
                poseTemp[2] = poseRaw.Translation.Z;
                poseTemp[3] = poseRaw.Rotation.X;
                poseTemp[4] = poseRaw.Rotation.Y;
                poseTemp[5] = poseRaw.Rotation.Z;


            }
            else
            {
                poseTemp[0] = Double.NaN;
                poseTemp[1] = Double.NaN;
                poseTemp[2] = Double.NaN;
                poseTemp[3] = Double.NaN;
                poseTemp[4] = Double.NaN;
                poseTemp[5] = Double.NaN;
            }

            PoseCameraFrame = poseTemp;
            standardKalmanFiltering(poseTemp);
            robustKalmanFiltering(poseTemp);
            addToBuffers();
        }

        private void standardKalmanFiltering(Vector<double> newPose)
        {
            if (!filterActivated[1]) return;

            if (!standardKalmanEnabled)
            {
                standardKalman.Initialize(newPose);
                PoseCameraFrameKf = newPose;
                standardKalmanEnabled = true;
            }
            else
            {
                Vector<double> _poseTemp = standardKalman.Estimate(newPose);
                PoseCameraFrameKf = _poseTemp;
            }
        }

        private void robustKalmanFiltering(Vector<double> newPose)
        {
            if (!filterActivated[2]) return;

            if (!robustKalmanEnabled)
            {
                robustKalman.Initialize(newPose);
                PoseCameraFrameRkf = newPose;
                robustKalmanEnabled = true;
            }
            else
            {
                Vector<double> _poseTemp = robustKalman.Estimate(newPose);
                PoseCameraFrameRkf = _poseTemp;
            }
        }

        #endregion

        #region Memory & IO

        private void addToBuffers()
        {
            for (int i = 0; i < 3; i++)
            {
                if (filterActivated[i])
                {
                    lock(poseLock)
                    {
                        poseBuffers[i].Add(new PoseData(poseCameraFrame[i], time));
                    }
                }
            }
        }

        public void ExportBuffers(string[] path)
        {
            for (int i = 0; i < path.Length; i++)
            {
                if (filterActivated[i])
                {
                    var poseDataList = poseBuffers[i].Memory.ToList();
                    Csv.WriteCsv(path[i], poseDataList);
                }
            }
        }

        public void ExportBuffersLast(string[] path, bool append=true, bool header=false) 
        {
            lock (poseLock)
            {
                time = stopWatch.ElapsedMilliseconds / 1000;
                for (int i = 0; i < path.Length; i++)
                {
                    if (filterActivated[i])
                    {
                        PoseData poseData;
                        if (header)
                        {
                            poseData = new PoseData(new double[6], 0);
                        }
                        else { poseData = poseBuffers[i].Memory.Last(); }
                        Csv.AppendCsv(path[i], poseData, append, header);
                    }
                }
            }
        }

        #endregion

        #region Event Hanlders

        private void ApiDisconnected()
        {
            ResetApi();
            ApiManager.Disconnected -= ApiDisconnected;
        }

        private void TrackerPoseChanged()
        {

        }

        private void ModelChanged()
        {

        }

        private void ModelDetectionStarted()
        {

        }

        private void ModelDetectionStopped(ITrackingModel trackingModel)
        {

        }

        private void TrackingStarted()
        {
            stopWatch.Start();
        }

        private void TrackingStopped()
        {
            stopWatch.Stop();
        }

        private void TrackingDataReady()
        {
            var trackingModel = iTrackingModelList[0];
            ProcessPose3d(trackingModel);
        }

        private void ThreadDataReady()
        {

        }

        #endregion

    }
}
