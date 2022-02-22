Imports FRRobot

Public Module PCDK

    ' Connection
    Public Robot As FRCRobot = New FRCRobot
    Public RobotConnected As Boolean = False
    Public RobotName As String

    'DPM
    Private objVarChannels(6) As FRRobot.FRCVar
    Private channels(6) As String

    'IO
    Private digIns(10) As FRCDigitalIOSignal
    Private digOuts(10) As FRCDigitalIOSignal

    'TP program
    Public Program As FRCTPProgram
    Public WithEvents RobotTasks As FRCTasks
    Public WithEvents RobotTask As FRCTask

    'Read data
    Private objCurPos As FRCCurPosition
    Private objCurGrpPos As FRCCurGroupPosition
    Private objCurXYZWPR As FRCXyzWpr
    Private objCurJoint As FRCJoint

#Region "Connection"
    Public Sub Connect(ByVal ipText As String)

        Robot.Connect(ipText)
        RobotConnected = True
        RobotName = Robot.SysVariables.Item("$SCR_GRP[1].$ROBOT_MODEL")

    End Sub

    Public Sub Disconnect()

        Robot = Nothing
        RobotConnected = False

    End Sub
#End Region

#Region "DPM and IO"

    Public Sub SetupDPM(ByVal sch As Integer)

        Dim channel As String
        For i As Integer = 0 To 5
            channel = "$DPM_SCH[" + sch + "].$GRP[1].$OFS[" + Str(i) + "].$INI_OFS"
            objVarChannels(i) = Robot.SysVariables(channel)
        Next

    End Sub

    Public Sub SetupIO()

        For i As Integer = 1 To 10
            digIns(i) = Robot.IOTypes.Item(1).Signals.Item(i) ' only simulated inputs
            digOuts(i) = Robot.IOTypes.Item(2).Signals.Item(i)
        Next

    End Sub

#End Region

#Region "TP Program"

    Public Sub Run(ByVal programName As String)
        Program = Robot.Programs.Item(programName)
        Program.Run()
        RobotTask = Robot.Tasks.Item(Name:=programName)
    End Sub

    Public Function IsRunning() As Boolean
        If RobotTask.Status = FRETaskStatusConstants.frStatusRun Then
            Return True
        Else
            Return False
        End If
    End Function

    Public Function IsAborted() As Boolean
        If RobotTask.Status = FRETaskStatusConstants.frStatusAborted Then
            Return True
        Else
            Return False
        End If
    End Function


#End Region

#Region "Get Data"

    Public Sub SetReferenceFrame(ByVal frame As Boolean)

        objCurPos = Robot.CurPosition
        If Not frame Then
            objCurGrpPos = objCurPos.Group(1, FRECurPositionConstants.frWorldDisplayType)
        Else
            objCurGrpPos = objCurPos.Group(1, FRECurPositionConstants.frUserDisplayType)
        End If
        objCurXYZWPR = objCurGrpPos.Formats(FRETypeCodeConstants.frXyzWpr)

    End Sub

    Public Function GetPose() As Double()

        Dim xyzwpr(6) As Double
        xyzwpr(0) = objCurXYZWPR.X
        xyzwpr(1) = objCurXYZWPR.Y
        xyzwpr(2) = objCurXYZWPR.Z
        xyzwpr(3) = objCurXYZWPR.W
        xyzwpr(4) = objCurXYZWPR.P
        xyzwpr(5) = objCurXYZWPR.R
        Return xyzwpr

    End Function

    Public Sub SetupJoint()

        objCurJoint = Robot.CurPosition.Group(1, FRECurPositionConstants.frJointDisplayType).Formats(FRETypeCodeConstants.frJoint)

    End Sub

    Public Function GetJointPosition() As Double()

        Dim jointVector(6) As Double
        jointVector(0) = objCurJoint(1)
        jointVector(1) = objCurJoint(2)
        jointVector(2) = objCurJoint(3)
        jointVector(3) = objCurJoint(4)
        jointVector(4) = objCurJoint(5)
        jointVector(5) = objCurJoint(6)
        Return jointVector

    End Function

#End Region

#Region "Integration"

    Public Sub SetupAll()

    End Sub

#End Region

End Module
