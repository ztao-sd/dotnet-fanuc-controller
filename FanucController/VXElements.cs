using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using VXelementsApi;
using VXelementsApi.Tracker;
using VXelementsApi.VXtrack;
using VXelementsApi.Types;
using MathNet.Numerics;

namespace FanucController
{
    internal class VXelementsUtility
    {
        // Api interfaces
        private IVXelements iVXelements;
        private ITracker iTracker;
        private IVXtrack iVXtrack;
        private bool trackerAttached;
        private bool vxtrackAttached;

        // Invokers
        public VXeventInvoker modelDetectionStartedInvoker;
        public DetectModelStoppedEventInvoker modelDetectionStoppedInvoker;
        public VXeventInvoker modelChangedInvoker;
        public VXeventInvoker trackingDataReadyInvoker;
        public VXeventInvoker trackingStartedInvoker;
        public VXeventInvoker trackingStoppedInvoker;
        public VXeventInvoker trackerPoseChangedInvoker;
        public VXeventInvoker sequenceChangedInvoker;

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
        public Point3dFloat translationCameraFrame;
        public Point3dFloat rotationCameraFrame;

        // Event
        public event EventHandler dataReady;

        public VXelementsUtility()
        {
            this.trackerAttached = false;
            this.vxtrackAttached = false;
            this.modelDetectionStartedInvoker = new VXeventInvoker(new VXeventHandler(ModelDetectionStarted));
            this.modelDetectionStoppedInvoker = new DetectModelStoppedEventInvoker(new DetectModelStoppedEventHandler(ModelDetectionStopped));
            this.modelChangedInvoker = new VXeventInvoker(new VXeventHandler(ModelChanged));
            this.trackingDataReadyInvoker = new VXeventInvoker(new VXeventHandler(TrackingDataReady));
            this.trackingStartedInvoker = new VXeventInvoker(new VXeventHandler(TrackingStarted));
            this.trackingStoppedInvoker = new VXeventInvoker(new VXeventHandler(TrackingStopped));
            this.trackerPoseChangedInvoker = new VXeventInvoker(new VXeventHandler(TrackerPoseChanged));
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

        public void FanucTrackingInitV1(string modelPath=null)
        {

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

        }
    }
}
