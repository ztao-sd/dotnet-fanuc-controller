Imports FRRobot

Public Module PCDK

#Region "Declarations"

    ' Connection
    Public Robot As FRCRobot = New FRCRobot
    Public RobotConnected As Boolean = False
    Public RobotName As String

    'DPM
    Private objVarChannels(5) As FRRobot.FRCVar
    Private channels(5) As String

    'IO
    Private digIns(9) As FRCDigitalIOSignal
    Private digOuts(9) As FRCDigitalIOSignal

    'TP program
    Public Program As FRCTPProgram
    Public WithEvents RobotTasks As FRCTasks
    Public WithEvents RobotTask As FRCTask

    'Read data
    Private objCurPos As FRCCurPosition
    Private objCurGrpPos(1) As FRCCurGroupPosition
    Private objCurXYZWPR(1) As FRCXyzWpr
    Private objCurJoint As FRCJoint

#End Region

#Region "Connection"
    Public Sub Connect(ByVal ipText As String)

        Robot.Connect(ipText)
        RobotConnected = True
        RobotName = Robot.SysVariables.Item("$SCR_GRP[1].$ROBOT_MODEL").Value

    End Sub

    Public Sub Disconnect()

        Robot = Nothing
        RobotConnected = False

    End Sub
#End Region

#Region "DPM and IO"

    Public Sub SetupDPM(ByVal sch As Integer)

        Dim channel As String
        For i As Integer = 1 To 6
            channel = "$DPM_SCH[" & sch.ToString() & "].$GRP[1].$OFS[" & i.ToString() & "].$INI_OFS"
            objVarChannels(i - 1) = Robot.SysVariables(channel)
        Next

    End Sub

    Public Sub SetupIO()

        For i As Integer = 1 To 10
            digIns(i - 1) = Robot.IOTypes.Item(1).Signals.Item(i) ' only simulated inputs
            digOuts(i - 1) = Robot.IOTypes.Item(2).Signals.Item(i)
        Next

    End Sub

    Public Function GetDigitalOutput(ByVal i As Integer) As Boolean

        Return digOuts(i - 1).Value

    End Function

    Public Function GetDigitalInput(ByVal i As Integer) As Boolean

        Return digIns(i - 1).Value

    End Function

    Public Sub SetDigitalInput(ByVal i As Integer, ByVal switch As Boolean)

        digIns(i - 1).Value = switch

    End Sub

    Public Sub SetDigitalOutput(ByVal i As Integer, ByVal switch As Boolean)

        digOuts(i - 1).Value = switch

    End Sub

    Public Sub ApplyDPM(ByVal u As Double(), ByVal sch As Integer)

        Dim channel As String
        'Dim objVar1chan As FRRobot.FRCVar
        'Dim objVar2chan As FRRobot.FRCVar
        'Dim objVar3chan As FRRobot.FRCVar
        'Dim objVar4chan As FRRobot.FRCVar
        'Dim objVar5chan As FRRobot.FRCVar
        'Dim objVar6chan As FRRobot.FRCVar

        'Dim ChannelX As String = "$DPM_SCH[" & sch.ToString() & "].$GRP[1].$OFS[1].$INI_OFS"
        'Dim ChannelY As String = "$DPM_SCH[" & sch.ToString() & "].$GRP[1].$OFS[2].$INI_OFS"
        'Dim ChannelZ As String = "$DPM_SCH[" & sch.ToString() & "].$GRP[1].$OFS[3].$INI_OFS"
        'Dim ChannelW As String = "$DPM_SCH[" & sch.ToString() & "].$GRP[1].$OFS[4].$INI_OFS"
        'Dim ChannelP As String = "$DPM_SCH[" & sch.ToString() & "].$GRP[1].$OFS[5].$INI_OFS"
        'Dim ChannelR As String = "$DPM_SCH[" & sch.ToString() & "].$GRP[1].$OFS[6].$INI_OFS"

        'objVar1chan = Robot.SysVariables(ChannelX)
        'objVar1chan.Value = u(0)
        'objVar2chan = Robot.SysVariables(ChannelY)
        'objVar2chan.Value = u(1)
        'objVar3chan = Robot.SysVariables(ChannelZ)
        'objVar3chan.Value = u(2)
        'objVar4chan = Robot.SysVariables(ChannelW)
        'objVar4chan.Value = u(3)
        'objVar5chan = Robot.SysVariables(ChannelP)
        'objVar5chan.Value = u(4)
        'objVar6chan = Robot.SysVariables(ChannelR)
        'objVar6chan.Value = u(5)

        For i As Integer = 0 To 5
            channel = "$DPM_SCH[" & sch.ToString() & "].$GRP[1].$OFS[" & (i + 1).ToString() & "].$INI_OFS"
            objVarChannels(i) = Robot.SysVariables(channel)
            objVarChannels(i).Value = u(i)
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

    Public Sub SetReferenceFrame()

        objCurPos = Robot.CurPosition
        objCurGrpPos(0) = objCurPos.Group(1, FRECurPositionConstants.frWorldDisplayType)
        objCurGrpPos(1) = objCurPos.Group(1, FRECurPositionConstants.frUserDisplayType)
        objCurXYZWPR(0) = objCurGrpPos(0).Formats(FRETypeCodeConstants.frXyzWpr)
        objCurXYZWPR(1) = objCurGrpPos(1).Formats(FRETypeCodeConstants.frXyzWpr)

    End Sub

    Public Function GetPoseWF() As Double()

        Dim xyzwpr(5) As Double
        xyzwpr(0) = objCurXYZWPR(0).X
        xyzwpr(1) = objCurXYZWPR(0).Y
        xyzwpr(2) = objCurXYZWPR(0).Z
        xyzwpr(3) = objCurXYZWPR(0).W
        xyzwpr(4) = objCurXYZWPR(0).P
        xyzwpr(5) = objCurXYZWPR(0).R
        Return xyzwpr

    End Function

    Public Function GetPoseUF() As Double()

        Dim xyzwpr(5) As Double
        xyzwpr(0) = objCurXYZWPR(1).X
        xyzwpr(1) = objCurXYZWPR(1).Y
        xyzwpr(2) = objCurXYZWPR(1).Z
        xyzwpr(3) = objCurXYZWPR(1).W
        xyzwpr(4) = objCurXYZWPR(1).P
        xyzwpr(5) = objCurXYZWPR(1).R
        Return xyzwpr

    End Function

    Public Sub SetupJoint()

        objCurJoint = Robot.CurPosition.Group(1, FRECurPositionConstants.frJointDisplayType).Formats(FRETypeCodeConstants.frJoint)

    End Sub

    Public Function GetJointPosition() As Double()

        Dim jointVector(5) As Double
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

    Public Sub SetupAll(ByVal sch As Integer)

        SetReferenceFrame()
        SetupDPM(sch)

    End Sub

#End Region

End Module
