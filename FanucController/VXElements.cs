using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using VXelementsApi;
using VXelementsApi.Tracker;
using VXelementsApi.VXtrack;
using VXelementsApi.Types;
using MathNet.Numerics;

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

        // Pose data
        public IPose3d poseCameraFrame;
        public Point3dDouble translationCameraFrame;
        public Point3dDouble rotationCameraFrame;

        // Event
        public event EventHandler DataReady;

        // Thread 
        private Thread eventThread;

        public VXelementsUtility()
        {
            // Flags
            this.trackerAttached = false;
            this.vxtrackAttached = false;
            // Invoker
            this.modelDetectionStartedInvoker = new VXeventInvoker(new VXeventHandler(ModelDetectionStarted));
            this.modelDetectionStoppedInvoker = new DetectModelStoppedEventInvoker(new DetectModelStoppedEventHandler(ModelDetectionStopped));
            this.modelChangedInvoker = new VXeventInvoker(new VXeventHandler(ModelChanged));
            this.trackingDataReadyInvoker = new VXeventInvoker(new VXeventHandler(TrackingDataReady));
            this.trackingStartedInvoker = new VXeventInvoker(new VXeventHandler(TrackingStarted));
            this.trackingStoppedInvoker = new VXeventInvoker(new VXeventHandler(TrackingStopped));
            this.trackerPoseChangedInvoker = new VXeventInvoker(new VXeventHandler(TrackerPoseChanged));
            // Thread
            this.eventThread = new Thread(new ThreadStart(ThreadDataReady));
            this.eventThread.IsBackground = true;
            // Tracking model list
            this.iTrackingModelList = new List<ITrackingModel>();
            // Tracking sequence list
            this.iTrackingSequenceList = new List<ITrackingSequence>();
        }

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
            this.iVXelements = null;
            this.iTracker = null;
            this.iVXtrack=null;
            this.vxtrackAttached = false;
            this.trackerAttached = false;
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
                if (!this.trackerAttached)
                {
                    this.iTracker.TrackerPoseChangedEvent.AddEvent(trackerPoseChangedInvoker);
                    this.trackerAttached = true;
                }
                if (!this.vxtrackAttached)
                {
                    this.iVXtrack.DetectModelStartedEvent.AddEvent(modelDetectionStartedInvoker);
                    this.iVXtrack.ModelsChangedEvent.AddEvent(modelChangedInvoker);
                    this.iVXtrack.TrackingDataReadyEvent.AddEvent(trackingDataReadyInvoker);
                    this.iVXtrack.TrackingStartedEvent.AddEvent(trackingStartedInvoker);
                    this.iVXtrack.TrackingStoppedEvent.AddEvent(trackingStoppedInvoker);
                    this.iVXtrack.SequencesChangedEvent.AddEvent(sequenceChangedInvoker);
                    this.vxtrackAttached = true;
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
                if (!this.trackerAttached)
                {
                    this.iTracker.TrackerPoseChangedEvent.RemoveEvent(trackerPoseChangedInvoker);
                    this.trackerAttached = false;
                }
                if (!this.vxtrackAttached)
                {
                    this.iVXtrack.DetectModelStartedEvent.RemoveEvent(modelDetectionStartedInvoker);
                    this.iVXtrack.ModelsChangedEvent.RemoveEvent(modelChangedInvoker);
                    this.iVXtrack.TrackingDataReadyEvent.RemoveEvent(trackingDataReadyInvoker);
                    this.iVXtrack.TrackingStartedEvent.RemoveEvent(trackingStartedInvoker);
                    this.iVXtrack.TrackingStoppedEvent.RemoveEvent(trackingStoppedInvoker);
                    this.iVXtrack.SequencesChangedEvent.RemoveEvent(sequenceChangedInvoker);
                    this.vxtrackAttached = false;
                }
            }
            catch
            {
                ExitApi();
            }
            
        }

        public void ClearSequences()
        {
            this.iVXtrack.RemoveAllSequences();
        }

        public void ClearEntities()
        {
            this.iVXtrack.RemoveAllEntities();
        }

        public void ResetVXTrack()
        {
            ClearEntities();
            ClearSequences();
        }

        public void OpenPositioningTargets(string targetPath=null)
        {
            this.iVXelements.OpenPositioningTargets(targetPath);
        }

        public void ImportModel(string modelPath = null)
        {
            this.iTrackingModelList.Add(this.iVXtrack.ImportModel(modelPath));
        }

        public void SaveModel(string savePath, ITrackingModel model)
        {
            model.Save(savePath);
        }

        public void DetectModel()
        {
            this.iVXtrack.DetectModel();
        }

        public void CreateSequence(ITrackingEntity[] trackingEntity=null)
        {
            this.iTrackingSequenceList.Add(this.iVXtrack.CreateSequence(trackingEntity));
        }

        public void ExportSequence(ITrackingSequence trackingSequence, string exportPath=null)
        {
            trackingSequence.ExportToCsv(exportPath);
        }

        public void AddModelRelation(ITrackingSequence currentSequence, ITrackingModel observedModel, ITrackingModel referenceModel)
        {
            currentSequence.AddTrackingModelRelation(observedModel, referenceModel);   
        }

        public void StartTracking(ITrackingSequence trackingSequence=null)
        {
            this.iVXtrack.StartTracking(trackingSequence);
        }

        public void StopTracking()
        {
            this.iVXtrack.StopTracking();
        }

        public IPose3d GetLastPose(ITrackingEntity trackingEntity)
        {
            var currentSequence = this.iVXtrack.CurrentTrackingSequence;
            return currentSequence.GetLastPose(trackingEntity);
        }

        public void ShowGraphics()
        {
            this.iVXtrack.ShowGraphicsViewForm();
            this.iVXtrack.ShowTableViewForm();
            this.iVXtrack.ShowProjectionViewForm();
        }

        public void HideGraphics()
        {
            this.iVXtrack.HideGraphicsViewForm();
            this.iVXtrack.HideTableViewForm();
            this.iVXtrack.HideProjectionViewForm();
        }

        public void FanucTrackingInitV1(string modelPath=null)
        {

        }

        private void ProcessPose3d(ITrackingEntity trackingEntity)
        {
            this.poseCameraFrame = GetLastPose(trackingEntity);
            if (this.poseCameraFrame.Valid)
            {
                this.translationCameraFrame.X = this.poseCameraFrame.Translation.X;
                this.translationCameraFrame.Y = this.poseCameraFrame.Translation.Y;
                this.translationCameraFrame.Z = this.poseCameraFrame.Translation.Z;
                this.rotationCameraFrame.X = this.poseCameraFrame.Rotation.X;
                this.rotationCameraFrame.Y = this.poseCameraFrame.Rotation.Y;
                this.rotationCameraFrame.Z = this.poseCameraFrame.Rotation.Z;


            } 
            else
            {
                this.translationCameraFrame.X = Double.NaN;
                this.translationCameraFrame.Y = Double.NaN;
                this.translationCameraFrame.Z = Double.NaN;
                this.rotationCameraFrame.X = Double.NaN;
                this.rotationCameraFrame.Y = Double.NaN;
                this.rotationCameraFrame.Z = Double.NaN;
            }
        }

        /*
         * Event Handlers
         */

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

        }

        private void TrackingStopped()
        {

        }
        
        private void TrackingDataReady()
        {
            var trackingModel = this.iTrackingModelList[0];
            ProcessPose3d(trackingModel);
        }

        private void ThreadDataReady()
        {

        }
    }
}
