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

namespace FanucController
{
    public class VXelementsUtility
    {
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
        private Vector<double> pose;
        private Vector<double> poseCameraFrame;
        private Vector<double> poseCameraFrameKf;
        private Vector<double> poseCameraFrameRkf;
        public Vector<double> PoseCameraFrame
        {
            get
            {
                lock (poseLock) { return poseCameraFrame; }
            }
            set
            {
                lock (poseLock) { poseCameraFrame = value; }
            }
        }
        public Vector<double> PoseCameraFrameKf
        {
            get
            {
                lock (poseLock) { return poseCameraFrameKf; }
            }
            set
            {
                lock (poseLock) { poseCameraFrameKf = value; }
            }
        }
        public Vector<double> PoseCameraFrameRkf
        {
            get
            {
                lock (poseLock) { return poseCameraFrameRkf; }
            }
            set
            {
                lock (poseLock) { poseCameraFrameRkf = value; }
            }
        }

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

        public VXelementsUtility()
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
            pose = CreateVector.Dense<double>(6);
            poseCameraFrame = CreateVector.Dense<double>(6);
            poseCameraFrameKf = CreateVector.Dense<double>(6);
            poseCameraFrameRkf = CreateVector.Dense<double>(6);
            

            // Buffer
            poseBuffers = new Buffer<PoseData>[3];
            poseBuffers[0] = new Buffer<PoseData>(20000); // raw
            poseBuffers[1] = new Buffer<PoseData>(20000); // kf
            poseBuffers[2] = new Buffer<PoseData>(20000); // rkf

            // Stopwatch
            stopWatch = new Stopwatch();
            time = 0;
        }

        // ---------------------------------------------- Actions

        public void QuickConnect(string targetsPath, string modelPath)
        {
            ConnectApi();
            AttachToVXelementsEvents();
            OpenPositioningTargets(targetsPath);
            ImportModel(modelPath);
            CreateSequence(iTrackingModelList.ToArray());
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

        // ---------------------------------------------- Basic Functions

        public void ConnectApi()
        {
            try
            {
                if (!ApiManager.IsConnected)
                {
                    ApiManager.Connect();
                    ApiManager.Disconnected += ApiDisconnected;
                    iVXelements = ApiManager.VXelements;
                    iTracker = iVXelements.Tracker;
                    iVXtrack = iVXelements.VXtrack;
                }
                else
                {
                    iVXelements = ApiManager.VXelements;
                    iTracker = iVXelements.Tracker;
                    iVXtrack = iVXelements.VXtrack;
                }

            }
            catch (VXelementsException ex)
            {
                Console.WriteLine(ex.ToString());
            }
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
                    iVXelements.Exit();
                    ResetApi();
                }
            }
            catch { }
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
                ResetApi();
                ConnectApi();
                AttachToVXelementsEvents();
            }

        }

        public void DetachFromVXelementsEvents()
        {
            try
            {
                // Remove invoker (event handlers) to event
                if (!trackerAttached)
                {
                    iTracker.TrackerPoseChangedEvent.RemoveEvent(trackerPoseChangedInvoker);
                    trackerAttached = false;
                }
                if (!vxtrackAttached)
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
            iVXtrack.StartTracking(trackingSequence);
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

        // ---------------------------------------------- Data Processing

        private void ProcessPose3d(ITrackingEntity trackingEntity)
        {
            // Lock from access to poseCameraFrame
            lock (poseLock)
            {
                poseRaw = GetLastPose(trackingEntity);
                if (poseRaw.Valid)
                {
                    poseCameraFrame[0] = poseRaw.Translation.X;
                    poseCameraFrame[1] = poseRaw.Translation.Y;
                    poseCameraFrame[2] = poseRaw.Translation.Z;
                    poseCameraFrame[3] = poseRaw.Rotation.X;
                    poseCameraFrame[4] = poseRaw.Rotation.Y;
                    poseCameraFrame[5] = poseRaw.Rotation.Z;


                }
                else
                {
                    poseCameraFrame[0] = Double.NaN;
                    poseCameraFrame[1] = Double.NaN;
                    poseCameraFrame[2] = Double.NaN;
                    poseCameraFrame[3] = Double.NaN;
                    poseCameraFrame[4] = Double.NaN;
                    poseCameraFrame[5] = Double.NaN;
                }

                standardKalmanFiltering();
                robustKalmanFiltering();
                addToBuffers();
            }
            
        }

        private void standardKalmanFiltering()
        {
            if (!standardKalmanEnabled)
            {
                standardKalman.Initialize(poseCameraFrame);
                poseCameraFrameKf = poseCameraFrame;
            }
            else
            {
                standardKalman.Estimate(poseCameraFrame);
            }
        }

        private void robustKalmanFiltering()
        {
            if (!robustKalmanEnabled)
            {
                robustKalman.Initialize(poseCameraFrame);
                poseCameraFrameRkf = poseCameraFrame;
            }
            else
            {
                poseCameraFrameRkf = robustKalman.Estimate(poseCameraFrame);
            }
        }

        // ---------------------------------------------- Memory & IO

        private void addToBuffers()
        {
            time = stopWatch.ElapsedMilliseconds/1000;
            poseBuffers[0].Add(new PoseData(poseCameraFrame, time));
            poseBuffers[1].Add(new PoseData(poseCameraFrame, time));
            poseBuffers[2].Add(new PoseData(poseCameraFrame, time));
        }

        public void ExportBuffers(string[] path)
        {
            for (int i = 0; i < path.Length; i++)
            {
                var poseDataList = poseBuffers[i].Memory.ToList();
                Csv.WriteCsv(path[i], poseDataList);
            }
        }

        // ---------------------------------------------- Event Handlers

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
    }
}
