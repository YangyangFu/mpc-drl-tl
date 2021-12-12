within ;
package FiveZoneAir "Five zone air system supervisory control"

  model Guideline36
    "Variable air volume flow system with terminal reheat and five thermal zones"
    extends Modelica.Icons.Example;
    extends FiveZoneAir.VAVReheat.BaseClasses.PartialOpenLoop(flo(
        cor(T_start=273.15 + 24),
        eas(T_start=273.15 + 24),
        sou(T_start=273.15 + 24),
        wes(T_start=273.15 + 24),
        nor(T_start=273.15 + 24)));
    extends FiveZoneAir.VAVReheat.BaseClasses.EnergyMeterAirSide(
      eleCoiVAV(y=cor.terHea.Q1_flow + nor.terHea.Q1_flow + wes.terHea.Q1_flow +
            eas.terHea.Q1_flow + sou.terHea.Q1_flow),
      eleSupFan(y=fanSup.P),
      elePla(y=cooCoi.Q1_flow/cooCOP),
      gasBoi(y=-heaCoi.Q1_flow));
    extends FiveZoneAir.VAVReheat.BaseClasses.ZoneAirTemperatureDeviation(
        banDevSum(each uppThreshold=24.5 + 273.15, each lowThreshold=23.5 + 273.15));
    parameter Modelica.SIunits.VolumeFlowRate VPriSysMax_flow=m_flow_nominal/1.2
      "Maximum expected system primary airflow rate at design stage";
    parameter Modelica.SIunits.VolumeFlowRate minZonPriFlo[numZon]={
        mCor_flow_nominal,mSou_flow_nominal,mEas_flow_nominal,mNor_flow_nominal,
        mWes_flow_nominal}/1.2 "Minimum expected zone primary flow rate";
    parameter Modelica.SIunits.Time samplePeriod=120
      "Sample period of component, set to the same value as the trim and respond that process yPreSetReq";
    parameter Modelica.SIunits.PressureDifference dpDisRetMax=40
      "Maximum return fan discharge static pressure setpoint";

    Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVCor(
      V_flow_nominal=mCor_flow_nominal/1.2,
      AFlo=AFloCor,
      final samplePeriod=samplePeriod,
      VDisSetMin_flow=0.05*conVAVCor.V_flow_nominal,
      VDisConMin_flow=0.05*conVAVCor.V_flow_nominal,
      errTZonCoo_1=0.8,
      errTZonCoo_2=0.4)                "Controller for terminal unit corridor"
      annotation (Placement(transformation(extent={{530,32},{550,52}})));
    Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVSou(
      V_flow_nominal=mSou_flow_nominal/1.2,
      AFlo=AFloSou,
      final samplePeriod=samplePeriod,
      VDisSetMin_flow=0.05*conVAVSou.V_flow_nominal,
      VDisConMin_flow=0.05*conVAVSou.V_flow_nominal,
      errTZonCoo_1=0.8,
      errTZonCoo_2=0.4)                "Controller for terminal unit south"
      annotation (Placement(transformation(extent={{700,30},{720,50}})));
    Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVEas(
      V_flow_nominal=mEas_flow_nominal/1.2,
      AFlo=AFloEas,
      final samplePeriod=samplePeriod,
      VDisSetMin_flow=0.05*conVAVEas.V_flow_nominal,
      VDisConMin_flow=0.05*conVAVEas.V_flow_nominal,
      errTZonCoo_1=0.8,
      errTZonCoo_2=0.4)                "Controller for terminal unit east"
      annotation (Placement(transformation(extent={{880,30},{900,50}})));
    Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVNor(
      V_flow_nominal=mNor_flow_nominal/1.2,
      AFlo=AFloNor,
      final samplePeriod=samplePeriod,
      VDisSetMin_flow=0.05*conVAVNor.V_flow_nominal,
      VDisConMin_flow=0.05*conVAVNor.V_flow_nominal,
      errTZonCoo_1=0.8,
      errTZonCoo_2=0.4)                "Controller for terminal unit north"
      annotation (Placement(transformation(extent={{1040,30},{1060,50}})));
    Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVWes(
      V_flow_nominal=mWes_flow_nominal/1.2,
      AFlo=AFloWes,
      final samplePeriod=samplePeriod,
      VDisSetMin_flow=0.05*conVAVWes.V_flow_nominal,
      VDisConMin_flow=0.05*conVAVWes.V_flow_nominal,
      errTZonCoo_1=0.8,
      errTZonCoo_2=0.4)                "Controller for terminal unit west"
      annotation (Placement(transformation(extent={{1240,28},{1260,48}})));
    Modelica.Blocks.Routing.Multiplex5 TDis "Discharge air temperatures"
      annotation (Placement(transformation(extent={{220,270},{240,290}})));
    Modelica.Blocks.Routing.Multiplex5 VDis_flow
      "Air flow rate at the terminal boxes"
      annotation (Placement(transformation(extent={{220,230},{240,250}})));
    Buildings.Controls.OBC.CDL.Integers.MultiSum TZonResReq(nin=5)
      "Number of zone temperature requests"
      annotation (Placement(transformation(extent={{300,360},{320,380}})));
    Buildings.Controls.OBC.CDL.Integers.MultiSum PZonResReq(nin=5)
      "Number of zone pressure requests"
      annotation (Placement(transformation(extent={{300,330},{320,350}})));
    Buildings.Controls.OBC.CDL.Continuous.Sources.Constant yOutDam(k=1)
      "Outdoor air damper control signal"
      annotation (Placement(transformation(extent={{-40,-20},{-20,0}})));
    Buildings.Controls.OBC.CDL.Logical.Switch swiFreSta "Switch for freeze stat"
      annotation (Placement(transformation(extent={{60,-202},{80,-182}})));
    Buildings.Controls.OBC.CDL.Continuous.Sources.Constant freStaSetPoi1(
      final k=273.15 + 3) "Freeze stat for heating coil"
      annotation (Placement(transformation(extent={{-40,-96},{-20,-76}})));
    Buildings.Controls.OBC.CDL.Continuous.Sources.Constant yFreHeaCoi(final k=1)
      "Flow rate signal for heating coil when freeze stat is on"
      annotation (Placement(transformation(extent={{0,-192},{20,-172}})));
    Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.ModeAndSetPoints TZonSet[
      numZon](
      final TZonHeaOn=fill(THeaOn, numZon),
      final TZonHeaOff=fill(THeaOff, numZon),
      final TZonCooOff=fill(TCooOff, numZon)) "Zone setpoint temperature"
      annotation (Placement(transformation(extent={{60,300},{80,320}})));
    Buildings.Controls.OBC.CDL.Routing.BooleanReplicator booRep(
      final nout=numZon) "Replicate boolean input"
      annotation (Placement(transformation(extent={{-120,280},{-100,300}})));
    Buildings.Controls.OBC.CDL.Routing.RealReplicator reaRep(
      final nout=numZon)
      "Replicate real input"
      annotation (Placement(transformation(extent={{-120,320},{-100,340}})));
    FiveZoneAir.VAVReheat.Controls.ControllerOve conAHU(
      kMinOut=0.01,
      final pMaxSet=410,
      final yFanMin=yFanMin,
      final VPriSysMax_flow=VPriSysMax_flow,
      final peaSysPop=1.2*sum({0.05*AFlo[i] for i in 1:numZon}),
      kTSup=0.01,
      TiTSup=120) "AHU controller"
      annotation (Placement(transformation(extent={{340,514},{420,642}})));
    Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.OutdoorAirFlow.Zone
      zonOutAirSet[numZon](
      final AFlo=AFlo,
      final have_occSen=fill(false, numZon),
      final have_winSen=fill(false, numZon),
      final desZonPop={0.05*AFlo[i] for i in 1:numZon},
      final minZonPriFlo=minZonPriFlo)
      "Zone level calculation of the minimum outdoor airflow setpoint"
      annotation (Placement(transformation(extent={{220,580},{240,600}})));
    Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.OutdoorAirFlow.SumZone
      zonToSys(final numZon=numZon) "Sum up zone calculation output"
      annotation (Placement(transformation(extent={{280,570},{300,590}})));
    Buildings.Controls.OBC.CDL.Routing.RealReplicator reaRep1(final nout=numZon)
      "Replicate design uncorrected minimum outdoor airflow setpoint"
      annotation (Placement(transformation(extent={{460,580},{480,600}})));
    Buildings.Controls.OBC.CDL.Routing.BooleanReplicator booRep1(final nout=numZon)
      "Replicate signal whether the outdoor airflow is required"
      annotation (Placement(transformation(extent={{460,550},{480,570}})));

    Buildings.Controls.OBC.CDL.Routing.BooleanReplicator booRepSupFan(final nout=
          numZon) "Replicate boolean input"
      annotation (Placement(transformation(extent={{500,630},{520,650}})));
    Buildings.Utilities.Math.Min minyDam(nin=5)
      "Computes lowest zone damper position"
      annotation (Placement(transformation(extent={{1350,-100},{1370,-80}})));
    Buildings.Utilities.Math.Max maxyDam(nin=5)
      annotation (Placement(transformation(extent={{1350,-170},{1370,-150}})));
    Buildings.Utilities.IO.SignalExchange.Read TOutAir(
      description=" outdoor air temperature",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="K")) "Read the outdoor air temperature"
      annotation (Placement(transformation(extent={{-300,130},{-280,150}})));

    Buildings.Utilities.IO.SignalExchange.Read GHI(
      description=" global horizontal solar radiation",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="W/m2")) "Read the global horizontal solar radiation"
      annotation (Placement(transformation(extent={{-300,90},{-280,110}})));

    Buildings.Utilities.IO.SignalExchange.Read yDamMax(
      description=" maximum zone air damper position",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="1")) "Read the maximum zone air damper position"
      annotation (Placement(transformation(extent={{1380,-170},{1400,-150}})));

    Buildings.Utilities.IO.SignalExchange.Read yDamMin(
      description=" minimum zone air damper position",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="1")) "Read the minimum zone air damper position"
      annotation (Placement(transformation(extent={{1380,-100},{1400,-80}})));

    Buildings.Utilities.IO.SignalExchange.Read PHVAC(
      description="total HVAC power",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
      y(unit="W"))
      "Read the total HVAC power"
      annotation (Placement(transformation(extent={{1320,640},{1340,660}})));

    Buildings.Utilities.IO.SignalExchange.Read PGas(
      description="total gas power",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.GasPower,
      y(unit="W")) "Read the total gas power"
      annotation (Placement(transformation(extent={{1320,570},{1340,590}})));

    Buildings.Utilities.IO.SignalExchange.Read dtTZonAir(
      description=" total zone air temperature deviation",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.AirZoneTemperature,
      y(unit="K")) "Read the total zone air temperature deviation"
      annotation (Placement(transformation(extent={{1320,440},{1340,460}})));

    Buildings.Utilities.IO.SignalExchange.Read yFanSpe(
      description=" fan speed",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="1")) "Read the fan speed"
      annotation (Placement(transformation(extent={{300,-100},{320,-80}})));

  equation
    connect(fanSup.port_b, dpDisSupFan.port_a) annotation (Line(
        points={{320,-40},{320,0},{320,-10},{320,-10}},
        color={0,0,0},
        smooth=Smooth.None,
        pattern=LinePattern.Dot));
    connect(conVAVCor.TZon, TRooAir.y5[1]) annotation (Line(
        points={{528,42},{520,42},{520,162},{511,162}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(conVAVSou.TZon, TRooAir.y1[1]) annotation (Line(
        points={{698,40},{690,40},{690,40},{680,40},{680,178},{511,178}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(TRooAir.y2[1], conVAVEas.TZon) annotation (Line(
        points={{511,174},{868,174},{868,40},{878,40}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(TRooAir.y3[1], conVAVNor.TZon) annotation (Line(
        points={{511,170},{1028,170},{1028,40},{1038,40}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(TRooAir.y4[1], conVAVWes.TZon) annotation (Line(
        points={{511,166},{1220,166},{1220,38},{1238,38}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(conVAVCor.TDis, TSupCor.T) annotation (Line(points={{528,36},{522,36},
            {522,40},{514,40},{514,92},{569,92}}, color={0,0,127}));
    connect(TSupSou.T, conVAVSou.TDis) annotation (Line(points={{749,92},{688,92},
            {688,34},{698,34}}, color={0,0,127}));
    connect(TSupEas.T, conVAVEas.TDis) annotation (Line(points={{929,90},{872,90},
            {872,34},{878,34}}, color={0,0,127}));
    connect(TSupNor.T, conVAVNor.TDis) annotation (Line(points={{1089,94},{1032,94},
            {1032,34},{1038,34}},     color={0,0,127}));
    connect(TSupWes.T, conVAVWes.TDis) annotation (Line(points={{1289,90},{1228,90},
            {1228,32},{1238,32}},     color={0,0,127}));
    connect(cor.yVAV, conVAVCor.yDam) annotation (Line(points={{566,50},{556,50},{
            556,48},{552,48}}, color={0,0,127}));
    connect(cor.yVal, conVAVCor.yVal) annotation (Line(points={{566,34},{560,34},{
            560,43},{552,43}}, color={0,0,127}));
    connect(conVAVSou.yDam, sou.yVAV) annotation (Line(points={{722,46},{730,46},{
            730,48},{746,48}}, color={0,0,127}));
    connect(conVAVSou.yVal, sou.yVal) annotation (Line(points={{722,41},{732.5,41},
            {732.5,32},{746,32}}, color={0,0,127}));
    connect(conVAVEas.yVal, eas.yVal) annotation (Line(points={{902,41},{912.5,41},
            {912.5,32},{926,32}}, color={0,0,127}));
    connect(conVAVEas.yDam, eas.yVAV) annotation (Line(points={{902,46},{910,46},{
            910,48},{926,48}}, color={0,0,127}));
    connect(conVAVNor.yDam, nor.yVAV) annotation (Line(points={{1062,46},{1072.5,46},
            {1072.5,48},{1086,48}},     color={0,0,127}));
    connect(conVAVNor.yVal, nor.yVal) annotation (Line(points={{1062,41},{1072.5,41},
            {1072.5,32},{1086,32}},     color={0,0,127}));
    connect(conVAVWes.yVal, wes.yVal) annotation (Line(points={{1262,39},{1272.5,39},
            {1272.5,32},{1286,32}},     color={0,0,127}));
    connect(wes.yVAV, conVAVWes.yDam) annotation (Line(points={{1286,48},{1274,48},
            {1274,44},{1262,44}}, color={0,0,127}));
    connect(conVAVCor.yZonTemResReq, TZonResReq.u[1]) annotation (Line(points={{552,38},
            {554,38},{554,220},{280,220},{280,375.6},{298,375.6}},         color=
            {255,127,0}));
    connect(conVAVSou.yZonTemResReq, TZonResReq.u[2]) annotation (Line(points={{722,36},
            {726,36},{726,220},{280,220},{280,372.8},{298,372.8}},         color=
            {255,127,0}));
    connect(conVAVEas.yZonTemResReq, TZonResReq.u[3]) annotation (Line(points={{902,36},
            {904,36},{904,220},{280,220},{280,370},{298,370}},         color={255,
            127,0}));
    connect(conVAVNor.yZonTemResReq, TZonResReq.u[4]) annotation (Line(points={{1062,36},
            {1064,36},{1064,220},{280,220},{280,367.2},{298,367.2}},
          color={255,127,0}));
    connect(conVAVWes.yZonTemResReq, TZonResReq.u[5]) annotation (Line(points={{1262,34},
            {1266,34},{1266,220},{280,220},{280,364.4},{298,364.4}},
          color={255,127,0}));
    connect(conVAVCor.yZonPreResReq, PZonResReq.u[1]) annotation (Line(points={{552,34},
            {558,34},{558,214},{288,214},{288,345.6},{298,345.6}},         color=
            {255,127,0}));
    connect(conVAVSou.yZonPreResReq, PZonResReq.u[2]) annotation (Line(points={{722,32},
            {728,32},{728,214},{288,214},{288,342.8},{298,342.8}},         color=
            {255,127,0}));
    connect(conVAVEas.yZonPreResReq, PZonResReq.u[3]) annotation (Line(points={{902,32},
            {906,32},{906,214},{288,214},{288,340},{298,340}},         color={255,
            127,0}));
    connect(conVAVNor.yZonPreResReq, PZonResReq.u[4]) annotation (Line(points={{1062,32},
            {1066,32},{1066,214},{288,214},{288,337.2},{298,337.2}},
          color={255,127,0}));
    connect(conVAVWes.yZonPreResReq, PZonResReq.u[5]) annotation (Line(points={{1262,30},
            {1268,30},{1268,214},{288,214},{288,334.4},{298,334.4}},
          color={255,127,0}));
    connect(VSupCor_flow.V_flow, VDis_flow.u1[1]) annotation (Line(points={{569,130},
            {472,130},{472,206},{180,206},{180,250},{218,250}},      color={0,0,
            127}));
    connect(VSupSou_flow.V_flow, VDis_flow.u2[1]) annotation (Line(points={{749,130},
            {742,130},{742,206},{180,206},{180,245},{218,245}},      color={0,0,
            127}));
    connect(VSupEas_flow.V_flow, VDis_flow.u3[1]) annotation (Line(points={{929,128},
            {914,128},{914,206},{180,206},{180,240},{218,240}},      color={0,0,
            127}));
    connect(VSupNor_flow.V_flow, VDis_flow.u4[1]) annotation (Line(points={{1089,132},
            {1080,132},{1080,206},{180,206},{180,235},{218,235}},      color={0,0,
            127}));
    connect(VSupWes_flow.V_flow, VDis_flow.u5[1]) annotation (Line(points={{1289,128},
            {1284,128},{1284,206},{180,206},{180,230},{218,230}},      color={0,0,
            127}));
    connect(TSupCor.T, TDis.u1[1]) annotation (Line(points={{569,92},{466,92},{466,
            210},{176,210},{176,290},{218,290}},     color={0,0,127}));
    connect(TSupSou.T, TDis.u2[1]) annotation (Line(points={{749,92},{688,92},{688,
            210},{176,210},{176,285},{218,285}},                       color={0,0,
            127}));
    connect(TSupEas.T, TDis.u3[1]) annotation (Line(points={{929,90},{872,90},{872,
            210},{176,210},{176,280},{218,280}},     color={0,0,127}));
    connect(TSupNor.T, TDis.u4[1]) annotation (Line(points={{1089,94},{1032,94},{1032,
            210},{176,210},{176,275},{218,275}},      color={0,0,127}));
    connect(TSupWes.T, TDis.u5[1]) annotation (Line(points={{1289,90},{1228,90},{1228,
            210},{176,210},{176,270},{218,270}},      color={0,0,127}));
    connect(conVAVCor.VDis_flow, VSupCor_flow.V_flow) annotation (Line(points={{528,40},
            {522,40},{522,130},{569,130}}, color={0,0,127}));
    connect(VSupSou_flow.V_flow, conVAVSou.VDis_flow) annotation (Line(points={{749,130},
            {690,130},{690,38},{698,38}},      color={0,0,127}));
    connect(VSupEas_flow.V_flow, conVAVEas.VDis_flow) annotation (Line(points={{929,128},
            {874,128},{874,38},{878,38}},      color={0,0,127}));
    connect(VSupNor_flow.V_flow, conVAVNor.VDis_flow) annotation (Line(points={{1089,
            132},{1034,132},{1034,38},{1038,38}}, color={0,0,127}));
    connect(VSupWes_flow.V_flow, conVAVWes.VDis_flow) annotation (Line(points={{1289,
            128},{1230,128},{1230,36},{1238,36}}, color={0,0,127}));
    connect(TSup.T, conVAVCor.TSupAHU) annotation (Line(points={{340,-29},{340,-20},
            {514,-20},{514,34},{528,34}}, color={0,0,127}));
    connect(TSup.T, conVAVSou.TSupAHU) annotation (Line(points={{340,-29},{340,-20},
            {686,-20},{686,32},{698,32}}, color={0,0,127}));
    connect(TSup.T, conVAVEas.TSupAHU) annotation (Line(points={{340,-29},{340,-20},
            {864,-20},{864,32},{878,32}}, color={0,0,127}));
    connect(TSup.T, conVAVNor.TSupAHU) annotation (Line(points={{340,-29},{340,-20},
            {1028,-20},{1028,32},{1038,32}}, color={0,0,127}));
    connect(TSup.T, conVAVWes.TSupAHU) annotation (Line(points={{340,-29},{340,-20},
            {1224,-20},{1224,30},{1238,30}}, color={0,0,127}));
    connect(yOutDam.y, eco.yExh)
      annotation (Line(points={{-18,-10},{-3,-10},{-3,-34}}, color={0,0,127}));
    connect(swiFreSta.y, gaiHeaCoi.u) annotation (Line(points={{82,-192},{88,-192},
            {88,-210},{98,-210}}, color={0,0,127}));
    connect(freSta.y, swiFreSta.u2) annotation (Line(points={{22,-92},{40,-92},{40,
            -192},{58,-192}},    color={255,0,255}));
    connect(yFreHeaCoi.y, swiFreSta.u1) annotation (Line(points={{22,-182},{40,-182},
            {40,-184},{58,-184}}, color={0,0,127}));
    connect(TZonSet[1].yOpeMod, conVAVCor.uOpeMod) annotation (Line(points={{82,303},
            {130,303},{130,180},{420,180},{420,14},{520,14},{520,32},{528,32}},
          color={255,127,0}));
    connect(flo.TRooAir, TZonSet.TZon) annotation (Line(points={{1094.14,
            491.333},{1164,491.333},{1164,662},{46,662},{46,313},{58,313}},
                                                                   color={0,0,127}));
    connect(occSch.occupied, booRep.u) annotation (Line(points={{-297,-216},{-160,
            -216},{-160,290},{-122,290}}, color={255,0,255}));
    connect(occSch.tNexOcc, reaRep.u) annotation (Line(points={{-297,-204},{-180,
            -204},{-180,330},{-122,330}},
                                    color={0,0,127}));
    connect(reaRep.y, TZonSet.tNexOcc) annotation (Line(points={{-98,330},{-20,330},
            {-20,319},{58,319}}, color={0,0,127}));
    connect(booRep.y, TZonSet.uOcc) annotation (Line(points={{-98,290},{-20,290},{
            -20,316.025},{58,316.025}}, color={255,0,255}));
    connect(TZonSet[1].TZonHeaSet, conVAVCor.TZonHeaSet) annotation (Line(points={{82,310},
            {524,310},{524,52},{528,52}},          color={0,0,127}));
    connect(TZonSet[1].TZonCooSet, conVAVCor.TZonCooSet) annotation (Line(points={{82,317},
            {524,317},{524,50},{528,50}},          color={0,0,127}));
    connect(TZonSet[2].TZonHeaSet, conVAVSou.TZonHeaSet) annotation (Line(points={{82,310},
            {694,310},{694,50},{698,50}},          color={0,0,127}));
    connect(TZonSet[2].TZonCooSet, conVAVSou.TZonCooSet) annotation (Line(points={{82,317},
            {694,317},{694,48},{698,48}},          color={0,0,127}));
    connect(TZonSet[3].TZonHeaSet, conVAVEas.TZonHeaSet) annotation (Line(points={{82,310},
            {860,310},{860,50},{878,50}},          color={0,0,127}));
    connect(TZonSet[3].TZonCooSet, conVAVEas.TZonCooSet) annotation (Line(points={{82,317},
            {860,317},{860,48},{878,48}},          color={0,0,127}));
    connect(TZonSet[4].TZonCooSet, conVAVNor.TZonCooSet) annotation (Line(points={{82,317},
            {1020,317},{1020,48},{1038,48}},          color={0,0,127}));
    connect(TZonSet[4].TZonHeaSet, conVAVNor.TZonHeaSet) annotation (Line(points={{82,310},
            {1020,310},{1020,50},{1038,50}},          color={0,0,127}));
    connect(TZonSet[5].TZonCooSet, conVAVWes.TZonCooSet) annotation (Line(points={{82,317},
            {1200,317},{1200,46},{1238,46}},          color={0,0,127}));
    connect(TZonSet[5].TZonHeaSet, conVAVWes.TZonHeaSet) annotation (Line(points={{82,310},
            {1200,310},{1200,48},{1238,48}},          color={0,0,127}));
    connect(TZonSet[1].yOpeMod, conVAVSou.uOpeMod) annotation (Line(points={{82,303},
            {130,303},{130,180},{420,180},{420,14},{680,14},{680,30},{698,30}},
          color={255,127,0}));
    connect(TZonSet[1].yOpeMod, conVAVEas.uOpeMod) annotation (Line(points={{82,303},
            {130,303},{130,180},{420,180},{420,14},{860,14},{860,30},{878,30}},
          color={255,127,0}));
    connect(TZonSet[1].yOpeMod, conVAVNor.uOpeMod) annotation (Line(points={{82,303},
            {130,303},{130,180},{420,180},{420,14},{1020,14},{1020,30},{1038,30}},
          color={255,127,0}));
    connect(TZonSet[1].yOpeMod, conVAVWes.uOpeMod) annotation (Line(points={{82,303},
            {130,303},{130,180},{420,180},{420,14},{1220,14},{1220,28},{1238,28}},
          color={255,127,0}));
    connect(zonToSys.ySumDesZonPop, conAHU.sumDesZonPop) annotation (Line(points={{302,589},
            {308,589},{308,611.778},{336,611.778}},           color={0,0,127}));
    connect(zonToSys.VSumDesPopBreZon_flow, conAHU.VSumDesPopBreZon_flow)
      annotation (Line(points={{302,586},{310,586},{310,606.444},{336,606.444}},
          color={0,0,127}));
    connect(zonToSys.VSumDesAreBreZon_flow, conAHU.VSumDesAreBreZon_flow)
      annotation (Line(points={{302,583},{312,583},{312,601.111},{336,601.111}},
          color={0,0,127}));
    connect(zonToSys.yDesSysVenEff, conAHU.uDesSysVenEff) annotation (Line(points={{302,580},
            {314,580},{314,595.778},{336,595.778}},           color={0,0,127}));
    connect(zonToSys.VSumUncOutAir_flow, conAHU.VSumUncOutAir_flow) annotation (
        Line(points={{302,577},{316,577},{316,590.444},{336,590.444}}, color={0,0,
            127}));
    connect(zonToSys.VSumSysPriAir_flow, conAHU.VSumSysPriAir_flow) annotation (
        Line(points={{302,571},{318,571},{318,585.111},{336,585.111}}, color={0,0,
            127}));
    connect(zonToSys.uOutAirFra_max, conAHU.uOutAirFra_max) annotation (Line(
          points={{302,574},{320,574},{320,579.778},{336,579.778}}, color={0,0,127}));
    connect(zonOutAirSet.yDesZonPeaOcc, zonToSys.uDesZonPeaOcc) annotation (Line(
          points={{242,599},{270,599},{270,588},{278,588}},     color={0,0,127}));
    connect(zonOutAirSet.VDesPopBreZon_flow, zonToSys.VDesPopBreZon_flow)
      annotation (Line(points={{242,596},{268,596},{268,586},{278,586}},
                                                       color={0,0,127}));
    connect(zonOutAirSet.VDesAreBreZon_flow, zonToSys.VDesAreBreZon_flow)
      annotation (Line(points={{242,593},{266,593},{266,584},{278,584}},
          color={0,0,127}));
    connect(zonOutAirSet.yDesPriOutAirFra, zonToSys.uDesPriOutAirFra) annotation (
       Line(points={{242,590},{264,590},{264,578},{278,578}},     color={0,0,127}));
    connect(zonOutAirSet.VUncOutAir_flow, zonToSys.VUncOutAir_flow) annotation (
        Line(points={{242,587},{262,587},{262,576},{278,576}},     color={0,0,127}));
    connect(zonOutAirSet.yPriOutAirFra, zonToSys.uPriOutAirFra)
      annotation (Line(points={{242,584},{260,584},{260,574},{278,574}},
                                                       color={0,0,127}));
    connect(zonOutAirSet.VPriAir_flow, zonToSys.VPriAir_flow) annotation (Line(
          points={{242,581},{258,581},{258,572},{278,572}},     color={0,0,127}));
    connect(conAHU.yAveOutAirFraPlu, zonToSys.yAveOutAirFraPlu) annotation (Line(
          points={{424,588.667},{440,588.667},{440,468},{270,468},{270,582},{
            278,582}},
          color={0,0,127}));
    connect(conAHU.VDesUncOutAir_flow, reaRep1.u) annotation (Line(points={{424,
            599.333},{440,599.333},{440,590},{458,590}},
                                                color={0,0,127}));
    connect(reaRep1.y, zonOutAirSet.VUncOut_flow_nominal) annotation (Line(points={{482,590},
            {490,590},{490,464},{210,464},{210,581},{218,581}},          color={0,
            0,127}));
    connect(conAHU.yReqOutAir, booRep1.u) annotation (Line(points={{424,567.333},
            {444,567.333},{444,560},{458,560}},color={255,0,255}));
    connect(booRep1.y, zonOutAirSet.uReqOutAir) annotation (Line(points={{482,560},
            {496,560},{496,460},{206,460},{206,593},{218,593}}, color={255,0,255}));
    connect(flo.TRooAir, zonOutAirSet.TZon) annotation (Line(points={{1094.14,
            491.333},{1164,491.333},{1164,660},{210,660},{210,590},{218,590}},
                                                                      color={0,0,127}));
    connect(TDis.y, zonOutAirSet.TDis) annotation (Line(points={{241,280},{252,280},
            {252,340},{200,340},{200,587},{218,587}}, color={0,0,127}));
    connect(VDis_flow.y, zonOutAirSet.VDis_flow) annotation (Line(points={{241,240},
            {260,240},{260,346},{194,346},{194,584},{218,584}}, color={0,0,127}));
    connect(TZonSet[1].yOpeMod, conAHU.uOpeMod) annotation (Line(points={{82,303},
            {140,303},{140,533.556},{336,533.556}}, color={255,127,0}));
    connect(TZonResReq.y, conAHU.uZonTemResReq) annotation (Line(points={{322,370},
            {330,370},{330,528.222},{336,528.222}}, color={255,127,0}));
    connect(PZonResReq.y, conAHU.uZonPreResReq) annotation (Line(points={{322,340},
            {326,340},{326,522.889},{336,522.889}}, color={255,127,0}));
    connect(TZonSet[1].TZonHeaSet, conAHU.TZonHeaSet) annotation (Line(points={{82,310},
            {110,310},{110,638.444},{336,638.444}},      color={0,0,127}));
    connect(TZonSet[1].TZonCooSet, conAHU.TZonCooSet) annotation (Line(points={{82,317},
            {120,317},{120,633.111},{336,633.111}},      color={0,0,127}));
    connect(TOut.y, conAHU.TOut) annotation (Line(points={{-279,180},{-260,180},
            {-260,627.778},{336,627.778}},
                                     color={0,0,127}));
    connect(dpDisSupFan.p_rel, conAHU.ducStaPre) annotation (Line(points={{311,0},
            {160,0},{160,622.444},{336,622.444}}, color={0,0,127}));
    connect(TSup.T, conAHU.TSup) annotation (Line(points={{340,-29},{340,-20},{
            152,-20},{152,569.111},{336,569.111}},
                                               color={0,0,127}));
    connect(TRet.T, conAHU.TOutCut) annotation (Line(points={{100,151},{100,
            563.778},{336,563.778}},
                            color={0,0,127}));
    connect(VOut1.V_flow, conAHU.VOut_flow) annotation (Line(points={{-61,-20.9},
            {-61,547.778},{336,547.778}},color={0,0,127}));
    connect(TMix.T, conAHU.TMix) annotation (Line(points={{40,-29},{40,540.667},
            {336,540.667}},
                       color={0,0,127}));
    connect(conAHU.yOutDamPos, eco.yOut) annotation (Line(points={{424,524.667},
            {448,524.667},{448,36},{-10,36},{-10,-34}},
                                                   color={0,0,127}));
    connect(conAHU.yRetDamPos, eco.yRet) annotation (Line(points={{424,535.333},
            {442,535.333},{442,40},{-16.8,40},{-16.8,-34}},
                                                       color={0,0,127}));
    connect(conAHU.yCoo, gaiCooCoi.u) annotation (Line(points={{424,546},{452,546},
            {452,-274},{88,-274},{88,-248},{98,-248}}, color={0,0,127}));
    connect(conAHU.yHea, swiFreSta.u3) annotation (Line(points={{424,556.667},{
            458,556.667},{458,-280},{40,-280},{40,-200},{58,-200}},
                                                                color={0,0,127}));
    connect(conAHU.ySupFanSpe, fanSup.y) annotation (Line(points={{424,620.667},
            {432,620.667},{432,-14},{310,-14},{310,-28}},
                                                     color={0,0,127}));
    connect(cor.y_actual,conVAVCor.yDam_actual)  annotation (Line(points={{612,58},
            {620,58},{620,74},{518,74},{518,38},{528,38}}, color={0,0,127}));
    connect(sou.y_actual,conVAVSou.yDam_actual)  annotation (Line(points={{792,56},
            {800,56},{800,76},{684,76},{684,36},{698,36}}, color={0,0,127}));
    connect(eas.y_actual,conVAVEas.yDam_actual)  annotation (Line(points={{972,56},
            {980,56},{980,74},{864,74},{864,36},{878,36}}, color={0,0,127}));
    connect(nor.y_actual,conVAVNor.yDam_actual)  annotation (Line(points={{1132,
            56},{1140,56},{1140,74},{1024,74},{1024,36},{1038,36}}, color={0,0,
            127}));
    connect(wes.y_actual,conVAVWes.yDam_actual)  annotation (Line(points={{1332,
            56},{1340,56},{1340,74},{1224,74},{1224,34},{1238,34}}, color={0,0,
            127}));
    connect(flo.TRooAir, banDevSum.u1) annotation (Line(points={{1094.14,
            491.333},{1165.07,491.333},{1165.07,490},{1238,490}}, color={0,0,127}));
    connect(conAHU.ySupFan, booRepSupFan.u) annotation (Line(points={{424,
            631.333},{467,631.333},{467,640},{498,640}},
                                                color={255,0,255}));
    connect(booRepSupFan.y, banDevSum.uSupFan) annotation (Line(points={{522,640},
            {580,640},{580,656},{1154,656},{1154,484},{1238,484}},      color={
            255,0,255}));
    connect(gasBoi.y, gasTotInt.u)
      annotation (Line(points={{1241,544},{1318,544}}, color={0,0,127}));

    connect(cor.y_actual, minyDam.u[1]) annotation (Line(
        points={{612,58},{618,58},{618,-90},{1348,-90},{1348,-91.6}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(sou.y_actual, minyDam.u[2]) annotation (Line(
        points={{792,56},{798,56},{798,-88},{1348,-88},{1348,-90.8}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(eas.y_actual, minyDam.u[3]) annotation (Line(
        points={{972,56},{976,56},{976,-90},{1348,-90}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(nor.y_actual, minyDam.u[4]) annotation (Line(
        points={{1132,56},{1136,56},{1136,-92},{1348,-92},{1348,-89.2}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(wes.y_actual, minyDam.u[5]) annotation (Line(
        points={{1332,56},{1334,56},{1334,-88},{1348,-88},{1348,-88.4}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(maxyDam.u[1], cor.y_actual) annotation (Line(
        points={{1348,-161.6},{618,-161.6},{618,58},{612,58}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(sou.y_actual, maxyDam.u[2]) annotation (Line(
        points={{792,56},{800,56},{800,-158},{818,-158},{818,-160.8},{1348,-160.8}},
        color={0,0,127},
        pattern=LinePattern.Dash));

    connect(eas.y_actual, maxyDam.u[3]) annotation (Line(
        points={{972,56},{982,56},{982,-160},{1348,-160}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(nor.y_actual, maxyDam.u[4]) annotation (Line(
        points={{1132,56},{1132,-160},{1348,-160},{1348,-159.2}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(wes.y_actual, maxyDam.u[5]) annotation (Line(
        points={{1332,56},{1336,56},{1336,-160},{1348,-160},{1348,-158.4}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(TOutAir.u, weaBus.TDryBul) annotation (Line(points={{-302,140},{-310,
            140},{-310,180},{-320,180}}, color={0,0,127}), Text(
        string="%second",
        index=1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(GHI.u, weaBus.HGloHor) annotation (Line(points={{-302,100},{-314,100},
            {-314,180},{-320,180}}, color={0,0,127}), Text(
        string="%second",
        index=1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(maxyDam.y, yDamMax.u)
      annotation (Line(points={{1371,-160},{1378,-160}}, color={0,0,127}));
    connect(minyDam.y, yDamMin.u) annotation (Line(points={{1371,-90},{1374.5,-90},
            {1374.5,-90},{1378,-90}}, color={0,0,127}));
    connect(eleTot.y, PHVAC.u) annotation (Line(points={{1297.02,612},{1308,612},
            {1308,650},{1318,650}}, color={0,0,127}));
    connect(gasBoi.y, PGas.u) annotation (Line(points={{1241,544},{1280,544},{1280,
            580},{1318,580}}, color={0,0,127}));
    connect(dtTZonAir.u, TAirDev.y) annotation (Line(points={{1318,450},{1304,450},
            {1304,490},{1295.02,490}}, color={0,0,127}));
    connect(yFanSpe.u, fanSup.y) annotation (Line(points={{298,-90},{280,-90},{280,
            -10},{310,-10},{310,-28}}, color={0,0,127}));
    annotation (
      Diagram(coordinateSystem(preserveAspectRatio=false,extent={{-380,-320},{1400,
              680}})),
      Documentation(info="<html>
<p>
This model consist of an HVAC system, a building envelope model and a model
for air flow through building leakage and through open doors.
</p>
<p>
The HVAC system is a variable air volume (VAV) flow system with economizer
and a heating and cooling coil in the air handler unit. There is also a
reheat coil and an air damper in each of the five zone inlet branches.
</p>
<p>
See the model
<a href=\"modelica://Buildings.Examples.VAVReheat.BaseClasses.PartialOpenLoop\">
Buildings.Examples.VAVReheat.BaseClasses.PartialOpenLoop</a>
for a description of the HVAC system and the building envelope.
</p>
<p>
The control is based on ASHRAE Guideline 36, and implemented
using the sequences from the library
<a href=\"modelica://Buildings.Controls.OBC.ASHRAE.G36_PR1\">
Buildings.Controls.OBC.ASHRAE.G36_PR1</a> for
multi-zone VAV systems with economizer. The schematic diagram of the HVAC and control
sequence is shown in the figure below.
</p>
<p align=\"center\">
<img alt=\"image\" src=\"modelica://Buildings/Resources/Images/Examples/VAVReheat/vavControlSchematics.png\" border=\"1\"/>
</p>
<p>
A similar model but with a different control sequence can be found in
<a href=\"modelica://Buildings.Examples.VAVReheat.ASHRAE2006\">
Buildings.Examples.VAVReheat.ASHRAE2006</a>.
Note that this model, because of the frequent time sampling,
has longer computing time than
<a href=\"modelica://Buildings.Examples.VAVReheat.ASHRAE2006\">
Buildings.Examples.VAVReheat.ASHRAE2006</a>.
The reason is that the time integrator cannot make large steps
because it needs to set a time step each time the control samples
its input.
</p>
</html>",   revisions="<html>
<ul>
<li>
April 20, 2020, by Jianjun Hu:<br/>
Exported actual VAV damper position as the measured input data for terminal controller.<br/>
This is
for <a href=\"https://github.com/lbl-srg/modelica-buildings/issues/1873\">issue #1873</a>
</li>
<li>
March 20, 2020, by Jianjun Hu:<br/>
Replaced the AHU controller with reimplemented one. The new controller separates the
zone level calculation from the system level calculation and does not include
vector-valued calculations.<br/>
This is
for <a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/1829\">#1829</a>.
</li>
<li>
March 09, 2020, by Jianjun Hu:<br/>
Replaced the block that calculates operation mode and zone temperature setpoint,
with the new one that does not include vector-valued calculations.<br/>
This is
for <a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/1709\">#1709</a>.
</li>
<li>
May 19, 2016, by Michael Wetter:<br/>
Changed chilled water supply temperature to <i>6&deg;C</i>.
This is
for <a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/509\">#509</a>.
</li>
<li>
April 26, 2016, by Michael Wetter:<br/>
Changed controller for freeze protection as the old implementation closed
the outdoor air damper during summer.
This is
for <a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/511\">#511</a>.
</li>
<li>
January 22, 2016, by Michael Wetter:<br/>
Corrected type declaration of pressure difference.
This is
for <a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/404\">#404</a>.
</li>
<li>
September 24, 2015 by Michael Wetter:<br/>
Set default temperature for medium to avoid conflicting
start values for alias variables of the temperature
of the building and the ambient air.
This is for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/426\">issue 426</a>.
</li>
</ul>
</html>"),
      __Dymola_Commands(file=
            "modelica://Buildings/Resources/Scripts/Dymola/Examples/VAVReheat/Guideline36.mos"
          "Simulate and plot"),
      experiment(
        StartTime=17625600,
        StopTime=18230400,
        Tolerance=1e-06,
        __Dymola_Algorithm="Cvode"),
      Icon(coordinateSystem(extent={{-100,-100},{100,100}})));
  end Guideline36;

  model SystemCoolSeasonBaseline
    extends Guideline36(
        flo(
        cor(T_start=273.15 + 24),
        sou(T_start=273.15 + 24),
        eas(T_start=273.15 + 24),
        wes(T_start=273.15 + 24),
        nor(T_start=273.15 + 24)));
  end SystemCoolSeasonBaseline;

  model wrappedcool "Wrapped model for cooling case"
   // Input overwrite
   Modelica.Blocks.Interfaces.RealInput oveAct_TSupSet(unit="K", min=273.15+12, max=273.15+18) "Supply air temperature setpoint";
   Modelica.Blocks.Interfaces.RealInput oveAct_dpSet(unit="Pa") "Supply air dp setpoint";
   // Out read
   Modelica.Blocks.Interfaces.RealOutput TZoneAirDev_y(unit="K") = modCoo.dtTZonAir.y "Total zone air temperature deviation";
   Modelica.Blocks.Interfaces.RealOutput TOutAir_y(unit="K") = modCoo.TOutAir.y "Outdoor air temperature";
   Modelica.Blocks.Interfaces.RealOutput GHI_y(unit="W/m2") = modCoo.GHI.y "Global horizontal solar radiation";
   Modelica.Blocks.Interfaces.RealOutput PHVAC_y(unit="W") = modCoo.PHVAC.y "Total HVAC power";
   Modelica.Blocks.Interfaces.RealOutput yFanSpe_y(unit="1") = modCoo.yFanSpe.y "AHU fan speed";
   Modelica.Blocks.Interfaces.RealOutput yDamMax_y(unit="1") = modCoo.yDamMax.y "Maximum zone air damper position";
   Modelica.Blocks.Interfaces.RealOutput yDamMin_y(unit="1") = modCoo.yDamMin.y "Minimum zone air damper position";
   // Original model
    FiveZoneAir.SystemCoolSeasonBaseline modCoo(conAHU(
            supTemSetPoi(oveActTAirSup(uExt(y=oveAct_TSupSet), activate(y=true))),
            supFan(oveActdpAir(uExt(y=oveAct_dpSet), activate(y=true)))))
      "Original model with overwrites";

    annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
          coordinateSystem(preserveAspectRatio=false)));
  end wrappedcool;

  package VAVReheat "Variable air volume flow system with terminal reheat and five thermal zone"
    extends Modelica.Icons.ExamplesPackage;

    package Controls "Package with controller models"
        extends Modelica.Icons.VariantsPackage;
      expandable connector ControlBus
        "Empty control bus that is adapted to the signals connected to it"
        extends Modelica.Icons.SignalBus;
        annotation (
          Icon(coordinateSystem(preserveAspectRatio=true, extent={{-100,-100},{100,
                  100}}), graphics={Rectangle(
                extent={{-20,2},{22,-2}},
                lineColor={255,204,51},
                lineThickness=0.5)}),
          Documentation(info="<html>
<p>
This connector defines the <code>expandable connector</code> ControlBus that
is used to connect control signals.
Note, this connector is empty. When using it, the actual content is
constructed by the signals connected to this bus.
</p>
</html>"));
      end ControlBus;

      block CoolingCoilTemperatureSetpoint "Set point scheduler for cooling coil"
        extends Modelica.Blocks.Icons.Block;
        import FiveZoneAir.VAVReheat.Controls.OperationModes;
        parameter Modelica.SIunits.Temperature TCooOn=273.15+12
          "Cooling setpoint during on";
        parameter Modelica.SIunits.Temperature TCooOff=273.15+30
          "Cooling setpoint during off";
        Modelica.Blocks.Sources.RealExpression TSupSetCoo(
         y=if (mode.y == Integer(OperationModes.occupied) or
               mode.y == Integer(OperationModes.unoccupiedPreCool) or
               mode.y == Integer(OperationModes.safety)) then
                TCooOn else TCooOff) "Supply air temperature setpoint for cooling"
          annotation (Placement(transformation(extent={{-22,-50},{-2,-30}})));
        Modelica.Blocks.Interfaces.RealInput TSetHea(
          unit="K",
          displayUnit="degC") "Set point for heating coil"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Math.Add add
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));
        Modelica.Blocks.Sources.Constant dTMin(k=1)
          "Minimum offset for cooling coil setpoint"
          annotation (Placement(transformation(extent={{-20,10},{0,30}})));
        Modelica.Blocks.Math.Max max1
          annotation (Placement(transformation(extent={{60,-30},{80,-10}})));
        ControlBus controlBus
          annotation (Placement(transformation(extent={{-28,-90},{-8,-70}})));
        Modelica.Blocks.Routing.IntegerPassThrough mode
          annotation (Placement(transformation(extent={{40,-90},{60,-70}})));
        Modelica.Blocks.Interfaces.RealOutput TSet(
          unit="K",
          displayUnit="degC") "Temperature set point"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      equation
        connect(dTMin.y, add.u1) annotation (Line(
            points={{1,20},{10,20},{10,6},{18,6}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(add.y, max1.u1) annotation (Line(
            points={{41,6.10623e-16},{52,6.10623e-16},{52,-14},{58,-14}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(TSupSetCoo.y, max1.u2) annotation (Line(
            points={{-1,-40},{20,-40},{20,-26},{58,-26}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(controlBus.controlMode, mode.u) annotation (Line(
            points={{-18,-80},{38,-80}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(max1.y, TSet) annotation (Line(
            points={{81,-20},{86,-20},{86,0},{110,0},{110,5.55112e-16}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(TSetHea, add.u2) annotation (Line(
            points={{-120,1.11022e-15},{-52,1.11022e-15},{-52,-6},{18,-6}},
            color={0,0,127},
            smooth=Smooth.None));
        annotation ( Icon(graphics={
              Text(
                extent={{44,16},{90,-18}},
                lineColor={0,0,255},
                textString="TSetCoo"),
              Text(
                extent={{-88,22},{-20,-26}},
                lineColor={0,0,255},
                textString="TSetHea")}));
      end CoolingCoilTemperatureSetpoint;

      model DuctStaticPressureSetpoint "Computes the duct static pressure setpoint"
        extends Modelica.Blocks.Interfaces.MISO;
        parameter Modelica.SIunits.AbsolutePressure pMin(displayUnit="Pa") = 100
          "Minimum duct static pressure setpoint";
        parameter Modelica.SIunits.AbsolutePressure pMax(displayUnit="Pa") = 410
          "Maximum duct static pressure setpoint";
        parameter Real k=0.1 "Gain of controller";
        parameter Modelica.SIunits.Time Ti=60 "Time constant of integrator block";
        parameter Modelica.SIunits.Time Td=60 "Time constant of derivative block";
        parameter Modelica.Blocks.Types.SimpleController controllerType=Modelica.Blocks.Types.SimpleController.PID
          "Type of controller";
        Buildings.Controls.Continuous.LimPID limPID(
          controllerType=controllerType,
          k=k,
          Ti=Ti,
          Td=Td,
          initType=Modelica.Blocks.Types.InitPID.InitialState,
          reverseAction=true)
          annotation (Placement(transformation(extent={{-20,40},{0,60}})));
      protected
        Buildings.Utilities.Math.Max max(final nin=nin)
          annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
        Modelica.Blocks.Sources.Constant ySet(k=0.9)
          "Setpoint for maximum damper position"
          annotation (Placement(transformation(extent={{-60,40},{-40,60}})));
        Modelica.Blocks.Math.Add dp(final k2=-1) "Pressure difference"
          annotation (Placement(transformation(extent={{-20,-60},{0,-40}})));
        Modelica.Blocks.Sources.Constant pMaxSig(k=pMax)
          annotation (Placement(transformation(extent={{-60,-40},{-40,-20}})));
        Modelica.Blocks.Sources.Constant pMinSig(k=pMin)
          annotation (Placement(transformation(extent={{-60,-80},{-40,-60}})));
        Modelica.Blocks.Math.Add pSet "Pressure setpoint"
          annotation (Placement(transformation(extent={{60,-10},{80,10}})));
        Modelica.Blocks.Math.Product product
          annotation (Placement(transformation(extent={{20,10},{40,30}})));
      public
        Modelica.Blocks.Logical.Hysteresis hysteresis(uLow=283.15, uHigh=284.15)
          "Hysteresis to put fan on minimum revolution"
          annotation (Placement(transformation(extent={{-60,70},{-40,90}})));
        Modelica.Blocks.Interfaces.RealInput TOut "Outside air temperature"
          annotation (Placement(transformation(extent={{-140,60},{-100,100}})));
      protected
        Modelica.Blocks.Sources.Constant zero(k=0) "Zero output signal"
          annotation (Placement(transformation(extent={{20,42},{40,62}})));
      public
        Modelica.Blocks.Logical.Switch switch1
          annotation (Placement(transformation(extent={{60,50},{80,70}})));
      equation
        connect(max.u, u) annotation (Line(
            points={{-62,6.66134e-16},{-80,6.66134e-16},{-80,0},{-120,0},{-120,
                1.11022e-15}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(ySet.y, limPID.u_s) annotation (Line(
            points={{-39,50},{-22,50}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(max.y, limPID.u_m) annotation (Line(
            points={{-39,6.10623e-16},{-10,6.10623e-16},{-10,38}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(limPID.y, product.u1) annotation (Line(
            points={{1,50},{10,50},{10,26},{18,26}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(pMaxSig.y, dp.u1) annotation (Line(
            points={{-39,-30},{-32,-30},{-32,-44},{-22,-44}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(pMinSig.y, dp.u2) annotation (Line(
            points={{-39,-70},{-30,-70},{-30,-56},{-22,-56}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(dp.y, product.u2) annotation (Line(
            points={{1,-50},{10,-50},{10,14},{18,14}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(pMinSig.y, pSet.u2) annotation (Line(
            points={{-39,-70},{30,-70},{30,-6},{58,-6}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(pSet.y, y) annotation (Line(
            points={{81,6.10623e-16},{90.5,6.10623e-16},{90.5,5.55112e-16},{110,
                5.55112e-16}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(hysteresis.u, TOut) annotation (Line(
            points={{-62,80},{-120,80}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(product.y, switch1.u1) annotation (Line(
            points={{41,20},{50,20},{50,68},{58,68}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(zero.y, switch1.u3) annotation (Line(
            points={{41,52},{58,52}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(switch1.y, pSet.u1) annotation (Line(
            points={{81,60},{90,60},{90,20},{52,20},{52,6},{58,6}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(hysteresis.y, switch1.u2) annotation (Line(
            points={{-39,80},{46,80},{46,60},{58,60}},
            color={255,0,255},
            smooth=Smooth.None));
        annotation ( Icon(graphics={
              Text(
                extent={{-76,148},{50,-26}},
                textString="PSet",
                lineColor={0,0,127}),
              Text(
                extent={{-10,8},{44,-82}},
                lineColor={0,0,127},
                textString="%pMax"),
              Text(
                extent={{-16,-54},{48,-90}},
                lineColor={0,0,127},
                textString="%pMin")}));
      end DuctStaticPressureSetpoint;

      block Economizer "Controller for economizer"
        import FiveZoneAir.VAVReheat.Controls.OperationModes;
        parameter Modelica.SIunits.Temperature TFreSet=277.15
          "Lower limit for mixed air temperature for freeze protection";
        parameter Modelica.SIunits.TemperatureDifference dT(min=0.1) = 1
          "Temperture offset to activate economizer";
        parameter Modelica.SIunits.VolumeFlowRate VOut_flow_min(min=0)
          "Minimum outside air volume flow rate";

        Modelica.Blocks.Interfaces.RealInput TSupHeaSet
          "Supply temperature setpoint for heating" annotation (Placement(
              transformation(extent={{-140,-40},{-100,0}}), iconTransformation(extent=
                 {{-140,-40},{-100,0}})));
        Modelica.Blocks.Interfaces.RealInput TSupCooSet
          "Supply temperature setpoint for cooling"
          annotation (Placement(transformation(extent={{-140,-100},{-100,-60}})));
        Modelica.Blocks.Interfaces.RealInput TMix "Measured mixed air temperature"
          annotation (Placement(transformation(extent={{-140,80},{-100,120}}),
              iconTransformation(extent={{-140,80},{-100,120}})));
        ControlBus controlBus
          annotation (Placement(transformation(extent={{-50,50},{-30,70}})));
        Modelica.Blocks.Interfaces.RealInput VOut_flow
          "Measured outside air flow rate" annotation (Placement(transformation(
                extent={{-140,20},{-100,60}}), iconTransformation(extent={{-140,20},{
                  -100,60}})));
        Modelica.Blocks.Interfaces.RealInput TRet "Return air temperature"
          annotation (Placement(transformation(extent={{-140,140},{-100,180}}),
              iconTransformation(extent={{-140,140},{-100,180}})));
        Modelica.Blocks.Math.Gain gain(k=1/VOut_flow_min) "Normalize mass flow rate"
          annotation (Placement(transformation(extent={{-60,-60},{-40,-40}})));
        Buildings.Controls.Continuous.LimPID conV_flow(
          controllerType=Modelica.Blocks.Types.SimpleController.PI,
          k=k,
          Ti=Ti,
          yMax=0.995,
          yMin=0.005,
          Td=60) "Controller for outside air flow rate"
          annotation (Placement(transformation(extent={{-22,-20},{-2,0}})));
        Modelica.Blocks.Sources.Constant uni(k=1) "Unity signal"
          annotation (Placement(transformation(extent={{-60,-20},{-40,0}})));
        parameter Real k=1 "Gain of controller";
        parameter Modelica.SIunits.Time Ti "Time constant of integrator block";
        Modelica.Blocks.Interfaces.RealOutput yOA
          "Control signal for outside air damper" annotation (Placement(
              transformation(extent={{200,70},{220,90}}), iconTransformation(extent={
                  {200,70},{220,90}})));
        Modelica.Blocks.Routing.Extractor extractor(nin=6, index(start=1, fixed=true))
          "Extractor for control signal"
          annotation (Placement(transformation(extent={{120,-10},{140,10}})));
        Modelica.Blocks.Sources.Constant closed(k=0) "Signal to close OA damper"
          annotation (Placement(transformation(extent={{60,-90},{80,-70}})));
        Modelica.Blocks.Math.Max max
          "Takes bigger signal (OA damper opens for temp. control or for minimum outside air)"
          annotation (Placement(transformation(extent={{80,-10},{100,10}})));
        MixedAirTemperatureSetpoint TSetMix "Mixed air temperature setpoint"
          annotation (Placement(transformation(extent={{-20,64},{0,84}})));
        EconomizerTemperatureControl yOATMix(Ti=Ti, k=k)
          "Control signal for outdoor damper to track mixed air temperature setpoint"
          annotation (Placement(transformation(extent={{20,160},{40,180}})));
        Buildings.Controls.Continuous.LimPID yOATFre(
          k=k,
          Ti=Ti,
          Td=60,
          controllerType=Modelica.Blocks.Types.SimpleController.PI,
          yMax=1,
          yMin=0)
          "Control signal for outdoor damper to track freeze temperature setpoint"
          annotation (Placement(transformation(extent={{20,120},{40,140}})));
        Modelica.Blocks.Math.Min min
          "Takes bigger signal (OA damper opens for temp. control or for minimum outside air)"
          annotation (Placement(transformation(extent={{20,-20},{40,0}})));
        Modelica.Blocks.Sources.Constant TFre(k=TFreSet)
          "Setpoint for freeze protection"
          annotation (Placement(transformation(extent={{-20,100},{0,120}})));
        Modelica.Blocks.Interfaces.RealOutput yRet
          "Control signal for return air damper" annotation (Placement(transformation(
                extent={{200,-10},{220,10}}), iconTransformation(extent={{200,-10},{
                  220,10}})));
        Buildings.Controls.OBC.CDL.Continuous.AddParameter invSig(p=1, k=-1)
          "Invert control signal for interlocked damper"
          annotation (Placement(transformation(extent={{170,-10},{190,10}})));
      equation
        connect(VOut_flow, gain.u) annotation (Line(
            points={{-120,40},{-92,40},{-92,-50},{-62,-50}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(gain.y, conV_flow.u_m) annotation (Line(
            points={{-39,-50},{-12,-50},{-12,-22}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(uni.y, conV_flow.u_s) annotation (Line(
            points={{-39,-10},{-24,-10}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(controlBus.controlMode, extractor.index) annotation (Line(
            points={{-40,60},{-40,30},{60,30},{60,-30},{130,-30},{130,-12}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(max.y, extractor.u[Integer(OperationModes.occupied)]) annotation (
            Line(
            points={{101,0},{118,0}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(closed.y, extractor.u[Integer(OperationModes.unoccupiedOff)])
          annotation (Line(
            points={{81,-80},{110,-80},{110,0},{118,0}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(closed.y, extractor.u[Integer(OperationModes.unoccupiedNightSetBack)])
          annotation (Line(
            points={{81,-80},{110,-80},{110,0},{118,0}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(max.y, extractor.u[Integer(OperationModes.unoccupiedWarmUp)])
          annotation (Line(
            points={{101,0},{118,0}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(max.y, extractor.u[Integer(OperationModes.unoccupiedPreCool)])
          annotation (Line(
            points={{101,0},{118,0}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(closed.y, extractor.u[Integer(OperationModes.safety)]) annotation (
            Line(
            points={{81,-80},{110,-80},{110,0},{118,0}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(TSupHeaSet, TSetMix.TSupHeaSet) annotation (Line(
            points={{-120,-20},{-80,-20},{-80,80},{-22,80}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(TSupCooSet, TSetMix.TSupCooSet) annotation (Line(
            points={{-120,-80},{-72,-80},{-72,68},{-22,68}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(controlBus, TSetMix.controlBus) annotation (Line(
            points={{-40,60},{-13,60},{-13,81}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(yOATMix.TRet, TRet) annotation (Line(
            points={{18,176},{-90,176},{-90,160},{-120,160}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(controlBus.TOut, yOATMix.TOut) annotation (Line(
            points={{-40,60},{-40,172},{18,172}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(yOATMix.TMix, TMix) annotation (Line(
            points={{18,168},{-80,168},{-80,100},{-120,100}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(yOATMix.TMixSet, TSetMix.TSet) annotation (Line(
            points={{18,164},{6,164},{6,75},{1,75}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(yOATMix.yOA, max.u1) annotation (Line(
            points={{41,170},{74,170},{74,6},{78,6}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(min.u2, conV_flow.y) annotation (Line(
            points={{18,-16},{10,-16},{10,-10},{-1,-10}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(min.y, max.u2) annotation (Line(
            points={{41,-10},{60,-10},{60,-6},{78,-6}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(yOATFre.u_s, TMix) annotation (Line(points={{18,130},{-32,130},{-80,
                130},{-80,100},{-120,100}}, color={0,0,127}));
        connect(TFre.y, yOATFre.u_m) annotation (Line(points={{1,110},{14,110},{30,
                110},{30,118}}, color={0,0,127}));
        connect(yOATFre.y, min.u1) annotation (Line(points={{41,130},{48,130},{48,20},
                {10,20},{10,-4},{18,-4}}, color={0,0,127}));
        connect(yRet, invSig.y)
          annotation (Line(points={{210,0},{191,0}}, color={0,0,127}));
        connect(extractor.y, invSig.u)
          annotation (Line(points={{141,0},{168,0}}, color={0,0,127}));
        connect(extractor.y, yOA) annotation (Line(points={{141,0},{160,0},{160,80},{
                210,80}}, color={0,0,127}));
        annotation (
          Diagram(coordinateSystem(preserveAspectRatio=true, extent={{-100,-100},{200,
                  200}})),
          Icon(coordinateSystem(preserveAspectRatio=true, extent={{-100,-100},{200,
                  200}}), graphics={
              Rectangle(
                extent={{-100,200},{200,-100}},
                lineColor={0,0,0},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid),
              Text(
                extent={{-90,170},{-50,150}},
                lineColor={0,0,255},
                textString="TRet"),
              Text(
                extent={{-86,104},{-46,84}},
                lineColor={0,0,255},
                textString="TMix"),
              Text(
                extent={{-90,60},{-22,12}},
                lineColor={0,0,255},
                textString="VOut_flow"),
              Text(
                extent={{-90,-2},{-28,-40}},
                lineColor={0,0,255},
                textString="TSupHeaSet"),
              Text(
                extent={{-86,-58},{-24,-96}},
                lineColor={0,0,255},
                textString="TSupCooSet"),
              Text(
                extent={{138,96},{184,62}},
                lineColor={0,0,255},
                textString="yOA"),
              Text(
                extent={{140,20},{186,-14}},
                lineColor={0,0,255},
                textString="yRet")}),
          Documentation(info="<html>
<p>
This is a controller for an economizer with
that adjust the outside air dampers to meet the set point
for the mixing air, taking into account the minimum outside
air requirement and an override for freeze protection.
</p>
</html>",       revisions="<html>
<ul>
<li>
December 20, 2016, by Michael Wetter:<br/>
Added type conversion for enumeration when used as an array index.<br/>
This is for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/602\">#602</a>.
</li>
<li>
April 26, 2016, by Michael Wetter:<br/>
Changed controller for freeze protection as the old implementation closed
the outdoor air damper during summer.
This is
for <a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/511\">#511</a>.
</li>
</ul>
</html>"));
      end Economizer;

      block EconomizerTemperatureControl
        "Controller for economizer mixed air temperature"
        extends Modelica.Blocks.Icons.Block;
        import FiveZoneAir.VAVReheat.Controls.OperationModes;
        Buildings.Controls.Continuous.LimPID con(
          k=k,
          Ti=Ti,
          yMax=0.995,
          yMin=0.005,
          Td=60,
          controllerType=Modelica.Blocks.Types.SimpleController.PI)
          "Controller for mixed air temperature"
          annotation (Placement(transformation(extent={{60,-10},{80,10}})));
        parameter Real k=1 "Gain of controller";
        parameter Modelica.SIunits.Time Ti "Time constant of integrator block";
        Modelica.Blocks.Logical.Switch swi1
          annotation (Placement(transformation(extent={{0,-10},{20,10}})));
        Modelica.Blocks.Logical.Switch swi2
          annotation (Placement(transformation(extent={{0,-50},{20,-30}})));
        Modelica.Blocks.Interfaces.RealOutput yOA
          "Control signal for outside air damper"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));
        Modelica.Blocks.Interfaces.RealInput TRet "Return air temperature"
          annotation (Placement(transformation(extent={{-140,40},{-100,80}})));
        Modelica.Blocks.Interfaces.RealInput TOut "Outside air temperature"
          annotation (Placement(transformation(extent={{-140,0},{-100,40}})));
        Modelica.Blocks.Interfaces.RealInput TMix "Mixed air temperature"
          annotation (Placement(transformation(extent={{-140,-40},{-100,0}})));
        Modelica.Blocks.Interfaces.RealInput TMixSet
          "Setpoint for mixed air temperature"
          annotation (Placement(transformation(extent={{-140,-80},{-100,-40}})));
        Modelica.Blocks.Logical.Hysteresis hysConGai(uLow=-0.1, uHigh=0.1)
          "Hysteresis for control gain"
          annotation (Placement(transformation(extent={{-40,50},{-20,70}})));
        Modelica.Blocks.Math.Feedback feedback
          annotation (Placement(transformation(extent={{-70,50},{-50,70}})));
      equation
        connect(swi1.y, con.u_s)    annotation (Line(
            points={{21,6.10623e-16},{30,0},{40,1.27676e-15},{40,6.66134e-16},{58,
                6.66134e-16}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(swi2.y, con.u_m)    annotation (Line(
            points={{21,-40},{70,-40},{70,-12}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(con.y, yOA)    annotation (Line(
            points={{81,6.10623e-16},{90.5,6.10623e-16},{90.5,5.55112e-16},{110,
                5.55112e-16}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(swi1.u1, TMix) annotation (Line(
            points={{-2,8},{-80,8},{-80,-20},{-120,-20}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(swi2.u3, TMix) annotation (Line(
            points={{-2,-48},{-80,-48},{-80,-20},{-120,-20}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(swi1.u3, TMixSet) annotation (Line(
            points={{-2,-8},{-60,-8},{-60,-60},{-120,-60}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(swi2.u1, TMixSet) annotation (Line(
            points={{-2,-32},{-60,-32},{-60,-60},{-120,-60}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(feedback.u1, TRet) annotation (Line(points={{-68,60},{-68,60},{-88,60},
                {-88,60},{-120,60}}, color={0,0,127}));
        connect(TOut, feedback.u2)
          annotation (Line(points={{-120,20},{-60,20},{-60,52}}, color={0,0,127}));
        connect(feedback.y, hysConGai.u) annotation (Line(points={{-51,60},{-48,60},{
                -46,60},{-42,60}}, color={0,0,127}));
        connect(hysConGai.y, swi2.u2) annotation (Line(points={{-19,60},{-12,60},{-12,
                -40},{-2,-40}}, color={255,0,255}));
        connect(hysConGai.y, swi1.u2) annotation (Line(points={{-19,60},{-12,60},{-12,
                0},{-2,0}}, color={255,0,255}));
        annotation ( Icon(coordinateSystem(
                preserveAspectRatio=true, extent={{-100,-100},{100,100}}), graphics={
              Text(
                extent={{-92,78},{-66,50}},
                lineColor={0,0,127},
                textString="TRet"),
              Text(
                extent={{-88,34},{-62,6}},
                lineColor={0,0,127},
                textString="TOut"),
              Text(
                extent={{-86,-6},{-60,-34}},
                lineColor={0,0,127},
                textString="TMix"),
              Text(
                extent={{-84,-46},{-58,-74}},
                lineColor={0,0,127},
                textString="TMixSet"),
              Text(
                extent={{64,14},{90,-14}},
                lineColor={0,0,127},
                textString="yOA")}), Documentation(info="<html>
<p>
This controller outputs the control signal for the outside
air damper in order to regulate the mixed air temperature
<code>TMix</code>.
</p>
<h4>Implementation</h4>
<p>
If the control error <i>T<sub>mix,set</sub> - T<sub>mix</sub> &lt; 0</i>,
then more outside air is needed provided that <i>T<sub>out</sub> &lt; T<sub>ret</sub></i>,
where
<i>T<sub>out</sub></i> is the outside air temperature and
<i>T<sub>ret</sub></i> is the return air temperature.
However, if <i>T<sub>out</sub> &ge; T<sub>ret</sub></i>,
then less outside air is needed.
Hence, the control gain need to switch sign depending on this difference.
This is accomplished by taking the difference between these signals,
and then switching the input of the controller.
A hysteresis is used to avoid chattering, for example if
<code>TRet</code> has numerical noise in the simulation, or
measurement error in a real application.
</p>
</html>",       revisions="<html>
<ul>
<li>
April 1, 2016, by Michael Wetter:<br/>
Added hysteresis to avoid too many events that stall the simulation.
This is for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/502\">#502</a>.
</li>
<li>
March 8, 2013, by Michael Wetter:<br/>
First implementation.
</li>
</ul>
</html>"));
      end EconomizerTemperatureControl;

      block FanVFD "Controller for fan revolution"
        extends Modelica.Blocks.Interfaces.SISO;
        import FiveZoneAir.VAVReheat.Controls.OperationModes;
        Buildings.Controls.Continuous.LimPID con(
          yMax=1,
          Td=60,
          yMin=r_N_min,
          k=k,
          Ti=Ti,
          controllerType=controllerType,
          reset=Buildings.Types.Reset.Disabled)
                                         "Controller"
          annotation (Placement(transformation(extent={{-20,20},{0,40}})));
        Modelica.Blocks.Math.Gain gaiMea(k=1/xSet_nominal)
          "Gain to normalize measurement signal"
          annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
        parameter Real xSet_nominal "Nominal setpoint (used for normalization)";
        Modelica.Blocks.Sources.Constant off(k=0) "Off signal"
          annotation (Placement(transformation(extent={{-60,-60},{-40,-40}})));
        Modelica.Blocks.Math.Gain gaiSet(k=1/xSet_nominal)
          "Gain to normalize setpoint signal"
          annotation (Placement(transformation(extent={{-60,20},{-40,40}})));
        Modelica.Blocks.Interfaces.RealInput u_m
          "Connector of measurement input signal" annotation (Placement(
              transformation(
              extent={{-20,-20},{20,20}},
              rotation=90,
              origin={0,-120})));
        parameter Real r_N_min=0.01 "Minimum normalized fan speed";
        parameter Modelica.Blocks.Types.Init initType=Modelica.Blocks.Types.Init.NoInit
          "Type of initialization (1: no init, 2: steady state, 3/4: initial output)";
        parameter Real y_start=0 "Initial or guess value of output (= state)";

        parameter Modelica.Blocks.Types.SimpleController
          controllerType=.Modelica.Blocks.Types.SimpleController.PI
          "Type of controller"
          annotation (Dialog(group="Setpoint tracking"));
        parameter Real k=0.5 "Gain of controller"
          annotation (Dialog(group="Setpoint tracking"));
        parameter Modelica.SIunits.Time Ti=15 "Time constant of integrator block"
          annotation (Dialog(group="Setpoint tracking"));

        Buildings.Controls.OBC.CDL.Logical.Switch swi
          annotation (Placement(transformation(extent={{40,-10},{60,10}})));
        Buildings.Controls.OBC.CDL.Interfaces.BooleanInput uFan
          "Set to true to enable the fan on"
          annotation (Placement(transformation(extent={{-140,40},{-100,80}})));
      equation
        connect(gaiMea.y, con.u_m) annotation (Line(
            points={{-39,6.10623e-16},{-10,6.10623e-16},{-10,18}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(gaiSet.y, con.u_s) annotation (Line(
            points={{-39,30},{-22,30}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(u_m, gaiMea.u) annotation (Line(
            points={{1.11022e-15,-120},{1.11022e-15,-92},{-80,-92},{-80,0},{-62,0},{
                -62,6.66134e-16}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(gaiSet.u, u) annotation (Line(
            points={{-62,30},{-90,30},{-90,1.11022e-15},{-120,1.11022e-15}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(con.y, swi.u1)
          annotation (Line(points={{1,30},{18,30},{18,8},{38,8}}, color={0,0,127}));
        connect(off.y, swi.u3) annotation (Line(points={{-39,-50},{20,-50},{20,-8},{
                38,-8}}, color={0,0,127}));
        connect(swi.u2, uFan) annotation (Line(points={{38,0},{12,0},{12,60},{-120,60}},
              color={255,0,255}));
        connect(swi.y, y) annotation (Line(points={{61,0},{110,0}}, color={0,0,127}));
        annotation ( Icon(graphics={Text(
                extent={{-90,-50},{96,-96}},
                lineColor={0,0,255},
                textString="r_N_min=%r_N_min")}), Documentation(revisions="<html>
<ul>
<li>
December 20, 2016, by Michael Wetter:<br/>
Added type conversion for enumeration when used as an array index.<br/>
This is for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/602\">#602</a>.
</li>
</ul>
</html>"));
      end FanVFD;

      model MixedAirTemperatureSetpoint
        "Mixed air temperature setpoint for economizer"
        extends Modelica.Blocks.Icons.Block;
        Modelica.Blocks.Routing.Extractor TSetMix(
          nin=6,
          index(start=2, fixed=true)) "Mixed air setpoint temperature extractor"
          annotation (Placement(transformation(extent={{60,0},{80,20}})));
        Modelica.Blocks.Sources.Constant off(k=273.15 + 13)
          "Setpoint temperature to close damper"
          annotation (Placement(transformation(extent={{-80,20},{-60,40}})));
        Buildings.Utilities.Math.Average ave(nin=2)
          annotation (Placement(transformation(extent={{-20,-70},{0,-50}})));
        Modelica.Blocks.Interfaces.RealInput TSupHeaSet
          "Supply temperature setpoint for heating"
          annotation (Placement(transformation(extent={{-140,40},{-100,80}}), iconTransformation(extent={{-140,40},{-100,80}})));
        Modelica.Blocks.Interfaces.RealInput TSupCooSet
          "Supply temperature setpoint for cooling"
          annotation (Placement(transformation(extent={{-140,-80},{-100,-40}})));
        Modelica.Blocks.Sources.Constant TPreCoo(k=273.15 + 13)
          "Setpoint during pre-cooling"
          annotation (Placement(transformation(extent={{-80,-20},{-60,0}})));
        ControlBus controlBus
          annotation (Placement(transformation(extent={{-40,60},{-20,80}})));
        Modelica.Blocks.Interfaces.RealOutput TSet "Mixed air temperature setpoint"
          annotation (Placement(transformation(extent={{100,0},{120,20}})));
        Modelica.Blocks.Routing.Multiplex2 multiplex2_1
          annotation (Placement(transformation(extent={{-60,-70},{-40,-50}})));
      equation
        connect(TSetMix.u[1], ave.y) annotation (Line(
            points={{58,8.33333},{14,8.33333},{14,-60},{1,-60}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(ave.y, TSetMix.u[1])     annotation (Line(
            points={{1,-60},{42,-60},{42,8.33333},{58,8.33333}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(off.y, TSetMix.u[2]) annotation (Line(
            points={{-59,30},{40,30},{40,12},{58,12},{58,9}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(off.y, TSetMix.u[3]) annotation (Line(
            points={{-59,30},{40,30},{40,9.66667},{58,9.66667}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(off.y, TSetMix.u[4]) annotation (Line(
            points={{-59,30},{9.5,30},{9.5,10.3333},{58,10.3333}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(TPreCoo.y, TSetMix.u[5]) annotation (Line(
            points={{-59,-10},{0,-10},{0,11},{58,11}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(off.y, TSetMix.u[6]) annotation (Line(
            points={{-59,30},{40,30},{40,11.6667},{58,11.6667}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(controlBus.controlMode, TSetMix.index) annotation (Line(
            points={{-30,70},{-30,-14},{70,-14},{70,-2}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(TSetMix.y, TSet) annotation (Line(
            points={{81,10},{110,10}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(multiplex2_1.y, ave.u) annotation (Line(
            points={{-39,-60},{-22,-60}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(TSupCooSet, multiplex2_1.u2[1]) annotation (Line(
            points={{-120,-60},{-90,-60},{-90,-66},{-62,-66}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(TSupHeaSet, multiplex2_1.u1[1]) annotation (Line(
            points={{-120,60},{-90,60},{-90,-54},{-62,-54}},
            color={0,0,127},
            smooth=Smooth.None));
      end MixedAirTemperatureSetpoint;

      model ModeSelector "Finite State Machine for the operational modes"
        Modelica.StateGraph.InitialStep initialStep(nIn=0)
          annotation (Placement(transformation(extent={{-80,20},{-60,40}})));
        Modelica.StateGraph.Transition start "Starts the system"
          annotation (Placement(transformation(extent={{-50,20},{-30,40}})));
        State unOccOff(
          mode=FiveZoneAir.VAVReheat.Controls.OperationModes.unoccupiedOff,
          nIn=3,
          nOut=4) "Unoccupied off mode, no coil protection"
          annotation (Placement(transformation(extent={{-20,20},{0,40}})));
        State unOccNigSetBac(
          nOut=2,
          mode=FiveZoneAir.VAVReheat.Controls.OperationModes.unoccupiedNightSetBack,
          nIn=1) "Unoccupied night set back"
          annotation (Placement(transformation(extent={{80,20},{100,40}})));
        Modelica.StateGraph.Transition t2(
          enableTimer=true,
          waitTime=60,
          condition=TRooMinErrHea.y > delTRooOnOff/2)
          annotation (Placement(transformation(extent={{28,20},{48,40}})));
        parameter Modelica.SIunits.TemperatureDifference delTRooOnOff(min=0.1)=1
          "Deadband in room temperature between occupied on and occupied off";
        parameter Modelica.SIunits.Temperature TRooSetHeaOcc=293.15
          "Set point for room air temperature during heating mode";
        parameter Modelica.SIunits.Temperature TRooSetCooOcc=299.15
          "Set point for room air temperature during cooling mode";
        parameter Modelica.SIunits.Temperature TSetHeaCoiOut=303.15
          "Set point for air outlet temperature at central heating coil";
        Modelica.StateGraph.Transition t1(condition=delTRooOnOff/2 < -TRooMinErrHea.y,
          enableTimer=true,
          waitTime=30*60)
          annotation (Placement(transformation(extent={{50,70},{30,90}})));
        inner Modelica.StateGraph.StateGraphRoot stateGraphRoot
          annotation (Placement(transformation(extent={{160,160},{180,180}})));
        ControlBus cb
          annotation (Placement(transformation(extent={{-168,130},{-148,150}}),
              iconTransformation(extent={{-176,124},{-124,176}})));
        Modelica.Blocks.Routing.RealPassThrough TRooSetHea
          "Current heating setpoint temperature"
          annotation (Placement(transformation(extent={{-80,170},{-60,190}})));
        State morWarUp(mode=FiveZoneAir.VAVReheat.Controls.OperationModes.unoccupiedWarmUp,
                                                                                  nIn=2,
          nOut=1) "Morning warm up"
          annotation (Placement(transformation(extent={{-40,-100},{-20,-80}})));
        Modelica.StateGraph.TransitionWithSignal t6(enableTimer=true, waitTime=60)
          annotation (Placement(transformation(extent={{-76,-100},{-56,-80}})));
        Modelica.Blocks.Logical.LessEqualThreshold occThrSho(threshold=1800)
          "Signal to allow transition into morning warmup"
          annotation (Placement(transformation(extent={{-140,-190},{-120,-170}})));
        Modelica.StateGraph.TransitionWithSignal t5
          annotation (Placement(transformation(extent={{118,20},{138,40}})));
        State occ(       mode=FiveZoneAir.VAVReheat.Controls.OperationModes.occupied,
                                                                            nIn=3)
          "Occupied mode"
          annotation (Placement(transformation(extent={{60,-100},{80,-80}})));
        Modelica.Blocks.Routing.RealPassThrough TRooMin
          annotation (Placement(transformation(extent={{-80,140},{-60,160}})));
        Modelica.Blocks.Math.Feedback TRooMinErrHea "Room control error for heating"
          annotation (Placement(transformation(extent={{-40,170},{-20,190}})));
        Modelica.StateGraph.Transition t3(condition=TRooMin.y + delTRooOnOff/2 >
              TRooSetHeaOcc or occupied.y)
          annotation (Placement(transformation(extent={{10,-100},{30,-80}})));
        Modelica.Blocks.Routing.BooleanPassThrough occupied
          "outputs true if building is occupied"
          annotation (Placement(transformation(extent={{-80,80},{-60,100}})));
        Modelica.StateGraph.TransitionWithSignal t4(enableTimer=false)
          annotation (Placement(transformation(extent={{118,120},{98,140}})));
        State morPreCoo(
          nIn=1,
          mode=FiveZoneAir.VAVReheat.Controls.OperationModes.unoccupiedPreCool,
          nOut=1) "Pre-cooling mode"
          annotation (Placement(transformation(extent={{-40,-140},{-20,-120}})));
        Modelica.StateGraph.Transition t7(condition=TRooMin.y - delTRooOnOff/2 <
              TRooSetCooOcc or occupied.y)
          annotation (Placement(transformation(extent={{10,-140},{30,-120}})));
        Modelica.Blocks.Logical.And and1
          annotation (Placement(transformation(extent={{-100,-200},{-80,-180}})));
        Modelica.Blocks.Routing.RealPassThrough TRooAve "Average room temperature"
          annotation (Placement(transformation(extent={{-80,110},{-60,130}})));
        Modelica.Blocks.Sources.BooleanExpression booleanExpression(y=TRooAve.y <
              TRooSetCooOcc)
          annotation (Placement(transformation(extent={{-198,-224},{-122,-200}})));
        PreCoolingStarter preCooSta(TRooSetCooOcc=TRooSetCooOcc)
          "Model to start pre-cooling"
          annotation (Placement(transformation(extent={{-140,-160},{-120,-140}})));
        Modelica.StateGraph.TransitionWithSignal t9
          annotation (Placement(transformation(extent={{-90,-140},{-70,-120}})));
        Modelica.Blocks.Logical.Not not1
          annotation (Placement(transformation(extent={{-48,-180},{-28,-160}})));
        Modelica.Blocks.Logical.And and2
          annotation (Placement(transformation(extent={{80,100},{100,120}})));
        Modelica.Blocks.Logical.Not not2
          annotation (Placement(transformation(extent={{0,100},{20,120}})));
        Modelica.StateGraph.TransitionWithSignal t8
          "changes to occupied in case precooling is deactivated"
          annotation (Placement(transformation(extent={{30,-30},{50,-10}})));
        Modelica.Blocks.MathInteger.Sum sum(nu=5)
          annotation (Placement(transformation(extent={{-186,134},{-174,146}})));
        Modelica.Blocks.Interfaces.BooleanOutput yFan
          "True if the fans are to be switched on"
          annotation (Placement(transformation(extent={{220,-10},{240,10}})));
        Modelica.Blocks.MathBoolean.Or or1(nu=4)
          annotation (Placement(transformation(extent={{184,-6},{196,6}})));
      equation
        connect(start.outPort, unOccOff.inPort[1]) annotation (Line(
            points={{-38.5,30},{-29.75,30},{-29.75,30.6667},{-21,30.6667}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(initialStep.outPort[1], start.inPort) annotation (Line(
            points={{-59.5,30},{-44,30}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(unOccOff.outPort[1], t2.inPort)         annotation (Line(
            points={{0.5,30.375},{8.25,30.375},{8.25,30},{34,30}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(t2.outPort, unOccNigSetBac.inPort[1])  annotation (Line(
            points={{39.5,30},{79,30}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(unOccNigSetBac.outPort[1], t1.inPort)   annotation (Line(
            points={{100.5,30.25},{112,30.25},{112,80},{44,80}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(t1.outPort, unOccOff.inPort[2])          annotation (Line(
            points={{38.5,80},{-30,80},{-30,30},{-21,30}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(cb.dTNexOcc, occThrSho.u)             annotation (Line(
            points={{-158,140},{-150,140},{-150,-180},{-142,-180}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(t6.outPort, morWarUp.inPort[1]) annotation (Line(
            points={{-64.5,-90},{-41,-90},{-41,-89.5}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(t5.outPort, morWarUp.inPort[2]) annotation (Line(
            points={{129.5,30},{140,30},{140,-60},{-48,-60},{-48,-90.5},{-41,-90.5}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(unOccNigSetBac.outPort[2], t5.inPort)
                                               annotation (Line(
            points={{100.5,29.75},{113.25,29.75},{113.25,30},{124,30}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(cb.TRooMin, TRooMin.u) annotation (Line(
            points={{-158,140},{-92,140},{-92,150},{-82,150}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(TRooSetHea.y, TRooMinErrHea.u1)
                                          annotation (Line(
            points={{-59,180},{-38,180}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(TRooMin.y, TRooMinErrHea.u2)
                                          annotation (Line(
            points={{-59,150},{-30,150},{-30,172}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(unOccOff.outPort[2], t6.inPort) annotation (Line(
            points={{0.5,30.125},{12,30.125},{12,-48},{-80,-48},{-80,-90},{-70,-90}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(morWarUp.outPort[1], t3.inPort) annotation (Line(
            points={{-19.5,-90},{16,-90}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(cb.occupied, occupied.u) annotation (Line(
            points={{-158,140},{-120,140},{-120,90},{-82,90}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(occ.outPort[1], t4.inPort) annotation (Line(
            points={{80.5,-90},{150,-90},{150,130},{112,130}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(t4.outPort, unOccOff.inPort[3]) annotation (Line(
            points={{106.5,130},{-30,130},{-30,29.3333},{-21,29.3333}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(occThrSho.y, and1.u1) annotation (Line(
            points={{-119,-180},{-110,-180},{-110,-190},{-102,-190}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(and1.y, t6.condition) annotation (Line(
            points={{-79,-190},{-66,-190},{-66,-102}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(and1.y, t5.condition) annotation (Line(
            points={{-79,-190},{128,-190},{128,18}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(cb.TRooAve, TRooAve.u) annotation (Line(
            points={{-158,140},{-100,140},{-100,120},{-82,120}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(booleanExpression.y, and1.u2) annotation (Line(
            points={{-118.2,-212},{-110.1,-212},{-110.1,-198},{-102,-198}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(preCooSta.y, t9.condition) annotation (Line(
            points={{-119,-150},{-80,-150},{-80,-142}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(t9.outPort, morPreCoo.inPort[1]) annotation (Line(
            points={{-78.5,-130},{-59.75,-130},{-59.75,-130},{-41,-130}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(unOccOff.outPort[3], t9.inPort) annotation (Line(
            points={{0.5,29.875},{12,29.875},{12,0},{-100,0},{-100,-130},{-84,-130}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(cb, preCooSta.controlBus) annotation (Line(
            points={{-158,140},{-150,140},{-150,-144},{-136.2,-144}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(morPreCoo.outPort[1], t7.inPort) annotation (Line(
            points={{-19.5,-130},{16,-130}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(t7.outPort, occ.inPort[2]) annotation (Line(
            points={{21.5,-130},{30,-130},{30,-128},{46,-128},{46,-90},{59,-90}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(t3.outPort, occ.inPort[1]) annotation (Line(
            points={{21.5,-90},{42,-90},{42,-89.3333},{59,-89.3333}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(occThrSho.y, not1.u) annotation (Line(
            points={{-119,-180},{-110,-180},{-110,-170},{-50,-170}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(not1.y, and2.u2) annotation (Line(
            points={{-27,-170},{144,-170},{144,84},{66,84},{66,102},{78,102}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(and2.y, t4.condition) annotation (Line(
            points={{101,110},{108,110},{108,118}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(occupied.y, not2.u) annotation (Line(
            points={{-59,90},{-20,90},{-20,110},{-2,110}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(not2.y, and2.u1) annotation (Line(
            points={{21,110},{78,110}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(cb.TRooSetHea, TRooSetHea.u) annotation (Line(
            points={{-158,140},{-92,140},{-92,180},{-82,180}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(t8.outPort, occ.inPort[3]) annotation (Line(
            points={{41.5,-20},{52,-20},{52,-90.6667},{59,-90.6667}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(unOccOff.outPort[4], t8.inPort) annotation (Line(
            points={{0.5,29.625},{12,29.625},{12,-20},{36,-20}},
            color={0,0,0},
            smooth=Smooth.None));
        connect(occupied.y, t8.condition) annotation (Line(
            points={{-59,90},{-50,90},{-50,-40},{40,-40},{40,-32}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(morPreCoo.y, sum.u[1]) annotation (Line(
            points={{-19,-136},{-8,-136},{-8,-68},{-192,-68},{-192,143.36},{-186,
                143.36}},
            color={255,127,0},
            smooth=Smooth.None));
        connect(morWarUp.y, sum.u[2]) annotation (Line(
            points={{-19,-96},{-8,-96},{-8,-68},{-192,-68},{-192,141.68},{-186,141.68}},
            color={255,127,0},
            smooth=Smooth.None));
        connect(occ.y, sum.u[3]) annotation (Line(
            points={{81,-96},{100,-96},{100,-108},{-192,-108},{-192,140},{-186,140}},
            color={255,127,0},
            smooth=Smooth.None));
        connect(unOccOff.y, sum.u[4]) annotation (Line(
            points={{1,24},{6,24},{6,8},{-192,8},{-192,138.32},{-186,138.32}},
            color={255,127,0},
            smooth=Smooth.None));
        connect(unOccNigSetBac.y, sum.u[5]) annotation (Line(
            points={{101,24},{112,24},{112,8},{-192,8},{-192,136.64},{-186,136.64}},
            color={255,127,0},
            smooth=Smooth.None));
        connect(sum.y, cb.controlMode) annotation (Line(
            points={{-173.1,140},{-158,140}},
            color={255,127,0},
            smooth=Smooth.None), Text(
            textString="%second",
            index=1,
            extent={{6,3},{6,3}}));
        connect(yFan, or1.y)
          annotation (Line(points={{230,0},{196.9,0}}, color={255,0,255}));
        connect(unOccNigSetBac.active, or1.u[1]) annotation (Line(points={{90,19},{90,
                3.15},{184,3.15}}, color={255,0,255}));
        connect(occ.active, or1.u[2]) annotation (Line(points={{70,-101},{70,-104},{
                168,-104},{168,1.05},{184,1.05}}, color={255,0,255}));
        connect(morWarUp.active, or1.u[3]) annotation (Line(points={{-30,-101},{-30,
                -112},{170,-112},{170,-1.05},{184,-1.05}}, color={255,0,255}));
        connect(morPreCoo.active, or1.u[4]) annotation (Line(points={{-30,-141},{-30,
                -150},{174,-150},{174,-3.15},{184,-3.15}}, color={255,0,255}));
        annotation (Diagram(coordinateSystem(preserveAspectRatio=true, extent={{-220,
                  -220},{220,220}})), Icon(coordinateSystem(
                preserveAspectRatio=true, extent={{-220,-220},{220,220}}), graphics={
                Rectangle(
                extent={{-200,200},{200,-200}},
                lineColor={0,0,0},
                fillPattern=FillPattern.Solid,
                fillColor={215,215,215}), Text(
                extent={{-176,80},{192,-84}},
                lineColor={0,0,255},
                textString="%name")}));
      end ModeSelector;

      type OperationModes = enumeration(
          occupied "Occupied",
          unoccupiedOff "Unoccupied off",
          unoccupiedNightSetBack "Unoccupied, night set back",
          unoccupiedWarmUp "Unoccupied, warm-up",
          unoccupiedPreCool "Unoccupied, pre-cool",
          safety "Safety (smoke, fire, etc.)") "Enumeration for modes of operation";
      block PreCoolingStarter "Outputs true when precooling should start"
        extends Modelica.Blocks.Interfaces.BooleanSignalSource;
        parameter Modelica.SIunits.Temperature TOutLim = 286.15
          "Limit for activating precooling";
        parameter Modelica.SIunits.Temperature TRooSetCooOcc
          "Set point for room air temperature during cooling mode";
        ControlBus controlBus
          annotation (Placement(transformation(extent={{-72,50},{-52,70}})));
        Modelica.Blocks.Logical.GreaterThreshold greater(threshold=TRooSetCooOcc)
          annotation (Placement(transformation(extent={{-40,0},{-20,20}})));
        Modelica.Blocks.Logical.LessThreshold greater2(threshold=1800)
          annotation (Placement(transformation(extent={{-40,-80},{-20,-60}})));
        Modelica.Blocks.Logical.LessThreshold greater1(threshold=TOutLim)
          annotation (Placement(transformation(extent={{-40,-40},{-20,-20}})));
        Modelica.Blocks.MathBoolean.And and3(nu=3)
          annotation (Placement(transformation(extent={{28,-6},{40,6}})));
      equation
        connect(controlBus.dTNexOcc, greater2.u) annotation (Line(
            points={{-62,60},{-54,60},{-54,-70},{-42,-70}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(controlBus.TRooAve, greater.u) annotation (Line(
            points={{-62,60},{-54,60},{-54,10},{-42,10}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(controlBus.TOut, greater1.u) annotation (Line(
            points={{-62,60},{-54,60},{-54,-30},{-42,-30}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(and3.y, y) annotation (Line(
            points={{40.9,0},{110,0}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(greater.y, and3.u[1]) annotation (Line(
            points={{-19,10},{6,10},{6,2.8},{28,2.8}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(greater1.y, and3.u[2]) annotation (Line(
            points={{-19,-30},{6,-30},{6,0},{28,0},{28,2.22045e-016}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(greater2.y, and3.u[3]) annotation (Line(
            points={{-19,-70},{12,-70},{12,-2.8},{28,-2.8}},
            color={255,0,255},
            smooth=Smooth.None));
      end PreCoolingStarter;

      block RoomTemperatureSetpoint "Set point scheduler for room temperature"
        extends Modelica.Blocks.Icons.Block;
        import FiveZoneAir.VAVReheat.Controls.OperationModes;
        parameter Modelica.SIunits.Temperature THeaOn=293.15
          "Heating setpoint during on";
        parameter Modelica.SIunits.Temperature THeaOff=285.15
          "Heating setpoint during off";
        parameter Modelica.SIunits.Temperature TCooOn=297.15
          "Cooling setpoint during on";
        parameter Modelica.SIunits.Temperature TCooOff=303.15
          "Cooling setpoint during off";
        ControlBus controlBus
          annotation (Placement(transformation(extent={{10,50},{30,70}})));
        Modelica.Blocks.Routing.IntegerPassThrough mode
          annotation (Placement(transformation(extent={{60,50},{80,70}})));
        Modelica.Blocks.Sources.RealExpression setPoiHea(
           y=if (mode.y == Integer(OperationModes.occupied) or mode.y == Integer(OperationModes.unoccupiedWarmUp)
               or mode.y == Integer(OperationModes.safety)) then THeaOn else THeaOff)
          annotation (Placement(transformation(extent={{-80,70},{-60,90}})));
        Modelica.Blocks.Sources.RealExpression setPoiCoo(
          y=if (mode.y == Integer(OperationModes.occupied) or
                mode.y == Integer(OperationModes.unoccupiedPreCool) or
                mode.y == Integer(OperationModes.safety)) then TCooOn else TCooOff)
          "Cooling setpoint"
          annotation (Placement(transformation(extent={{-80,30},{-60,50}})));
      equation
        connect(controlBus.controlMode,mode. u) annotation (Line(
            points={{20,60},{58,60}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%first",
            index=-1,
            extent={{-6,3},{-6,3}}));
        connect(setPoiHea.y, controlBus.TRooSetHea) annotation (Line(
            points={{-59,80},{20,80},{20,60}},
            color={0,0,127},
            smooth=Smooth.None), Text(
            textString="%second",
            index=1,
            extent={{6,3},{6,3}}));
        connect(setPoiCoo.y, controlBus.TRooSetCoo) annotation (Line(
            points={{-59,40},{20,40},{20,60}},
            color={0,0,127},
            smooth=Smooth.None), Text(
            textString="%second",
            index=1,
            extent={{6,3},{6,3}}));
        annotation (                                Icon(graphics={
              Text(
                extent={{-92,90},{-52,70}},
                lineColor={0,0,255},
                textString="TRet"),
              Text(
                extent={{-96,50},{-56,30}},
                lineColor={0,0,255},
                textString="TMix"),
              Text(
                extent={{-94,22},{-26,-26}},
                lineColor={0,0,255},
                textString="VOut_flow"),
              Text(
                extent={{-88,-22},{-26,-60}},
                lineColor={0,0,255},
                textString="TSupHeaSet"),
              Text(
                extent={{-86,-58},{-24,-96}},
                lineColor={0,0,255},
                textString="TSupCooSet"),
              Text(
                extent={{42,16},{88,-18}},
                lineColor={0,0,255},
                textString="yOA")}));
      end RoomTemperatureSetpoint;

      block RoomVAV "Controller for room VAV box"
        extends Modelica.Blocks.Icons.Block;

        parameter Real ratVFloMin=0.3
          "VAV box minimum airflow ratio to the cooling maximum flow rate, typically between 0.3 to 0.5";
        parameter Buildings.Controls.OBC.CDL.Types.SimpleController cooController=
            Buildings.Controls.OBC.CDL.Types.SimpleController.PI "Type of controller"
          annotation (Dialog(group="Cooling controller"));
        parameter Real kCoo=0.1 "Gain of controller"
          annotation (Dialog(group="Cooling controller"));
        parameter Modelica.SIunits.Time TiCoo=120 "Time constant of integrator block"
          annotation (Dialog(group="Cooling controller", enable=cooController==Buildings.Controls.OBC.CDL.Types.SimpleController.PI or
                                                                cooController==Buildings.Controls.OBC.CDL.Types.SimpleController.PID));
        parameter Modelica.SIunits.Time TdCoo=60 "Time constant of derivative block"
          annotation (Dialog(group="Cooling controller", enable=cooController==Buildings.Controls.OBC.CDL.Types.SimpleController.PD or
                                                                cooController==Buildings.Controls.OBC.CDL.Types.SimpleController.PID));
        parameter Buildings.Controls.OBC.CDL.Types.SimpleController heaController=
            Buildings.Controls.OBC.CDL.Types.SimpleController.PI "Type of controller"
          annotation (Dialog(group="Heating controller"));
        parameter Real kHea=0.1 "Gain of controller"
          annotation (Dialog(group="Heating controller"));
        parameter Modelica.SIunits.Time TiHea=120 "Time constant of integrator block"
          annotation (Dialog(group="Heating controller", enable=heaController==Buildings.Controls.OBC.CDL.Types.SimpleController.PI or
                                                                heaController==Buildings.Controls.OBC.CDL.Types.SimpleController.PID));
        parameter Modelica.SIunits.Time TdHea=60 "Time constant of derivative block"
          annotation (Dialog(group="Heating controller", enable=heaController==Buildings.Controls.OBC.CDL.Types.SimpleController.PD or
                                                                heaController==Buildings.Controls.OBC.CDL.Types.SimpleController.PID));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput TRooHeaSet(
          final quantity="ThermodynamicTemperature",
          final unit = "K",
          displayUnit = "degC")
          "Setpoint temperature for room for heating"
          annotation (Placement(transformation(extent={{-140,40},{-100,80}}),
              iconTransformation(extent={{-140,50},{-100,90}})));
        Buildings.Controls.OBC.CDL.Interfaces.RealInput TRooCooSet(
          final quantity="ThermodynamicTemperature",
          final unit = "K",
          displayUnit = "degC")
          "Setpoint temperature for room for cooling"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}}),
              iconTransformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Interfaces.RealInput TRoo(
          final quantity="ThermodynamicTemperature",
          final unit = "K",
          displayUnit = "degC")
          "Measured room temperature"
          annotation (Placement(transformation(extent={{-140,-90},{-100,-50}}),
              iconTransformation(extent={{-120,-80},{-100,-60}})));
        Modelica.Blocks.Interfaces.RealOutput yDam "Signal for VAV damper"
          annotation (Placement(transformation(extent={{100,-10},{120,10}}),
              iconTransformation(extent={{100,38},{120,58}})));
        Modelica.Blocks.Interfaces.RealOutput yVal "Signal for heating coil valve"
          annotation (Placement(transformation(extent={{100,-80},{120,-60}}),
              iconTransformation(extent={{100,-60},{120,-40}})));

        Buildings.Controls.OBC.CDL.Continuous.LimPID conHea(
          yMax=yMax,
          Td=TdHea,
          yMin=yMin,
          k=kHea,
          Ti=TiHea,
          controllerType=heaController,
          Ni=10)                        "Controller for heating"
          annotation (Placement(transformation(extent={{40,-80},{60,-60}})));
        Buildings.Controls.OBC.CDL.Continuous.LimPID conCoo(
          yMax=yMax,
          Td=TdCoo,
          k=kCoo,
          Ti=TiCoo,
          controllerType=cooController,
          yMin=yMin,
          reverseAction=true)
          "Controller for cooling (acts on damper)"
          annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
        Buildings.Controls.OBC.CDL.Continuous.Line reqFlo "Required flow rate"
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant cooMax(k=1)
          "Cooling maximum flow"
          annotation (Placement(transformation(extent={{-20,-50},{0,-30}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant minFlo(k=ratVFloMin)
          "VAV box minimum flow"
          annotation (Placement(transformation(extent={{-60,30},{-40,50}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant conOne(k=1)
          "Constant 1"
          annotation (Placement(transformation(extent={{-60,-50},{-40,-30}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant conZer(k=0)
          "Constant 0"
          annotation (Placement(transformation(extent={{-20,30},{0,50}})));

      protected
        parameter Real yMax=1 "Upper limit of PID control output";
        parameter Real yMin=0 "Lower limit of PID control output";

      equation
        connect(TRooCooSet, conCoo.u_s)
          annotation (Line(points={{-120,0},{-62,0}}, color={0,0,127}));
        connect(TRoo, conHea.u_m) annotation (Line(points={{-120,-70},{-80,-70},{-80,-90},
                {50,-90},{50,-82}},        color={0,0,127}));
        connect(TRooHeaSet, conHea.u_s) annotation (Line(points={{-120,60},{-70,60},{-70,
                -70},{38,-70}},      color={0,0,127}));
        connect(conHea.y, yVal)
          annotation (Line(points={{62,-70},{110,-70}},  color={0,0,127}));
        connect(conZer.y, reqFlo.x1)
          annotation (Line(points={{2,40},{10,40},{10,8},{18,8}}, color={0,0,127}));
        connect(minFlo.y, reqFlo.f1) annotation (Line(points={{-38,40},{-30,40},{-30,4},
                {18,4}}, color={0,0,127}));
        connect(cooMax.y, reqFlo.f2) annotation (Line(points={{2,-40},{10,-40},{10,-8},
                {18,-8}},color={0,0,127}));
        connect(conOne.y, reqFlo.x2) annotation (Line(points={{-38,-40},{-30,-40},{-30,
                -4},{18,-4}}, color={0,0,127}));
        connect(conCoo.y, reqFlo.u)
          annotation (Line(points={{-38,0},{18,0}}, color={0,0,127}));
        connect(TRoo, conCoo.u_m) annotation (Line(points={{-120,-70},{-80,-70},{-80,
                -20},{-50,-20},{-50,-12}}, color={0,0,127}));
        connect(reqFlo.y, yDam)
          annotation (Line(points={{42,0},{72,0},{72,0},{110,0}}, color={0,0,127}));

      annotation (
        defaultComponentName="terCon",
        Icon(coordinateSystem(extent={{-100,-100},{100,100}}),
                          graphics={
              Text(
                extent={{-100,-62},{-66,-76}},
                lineColor={0,0,127},
                textString="TRoo"),
              Text(
                extent={{64,-38},{92,-58}},
                lineColor={0,0,127},
                textString="yVal"),
              Text(
                extent={{56,62},{90,40}},
                lineColor={0,0,127},
                textString="yDam"),
              Text(
                extent={{-96,82},{-36,60}},
                lineColor={0,0,127},
                textString="TRooHeaSet"),
              Text(
                extent={{-96,10},{-36,-10}},
                lineColor={0,0,127},
                textString="TRooCooSet")}),
       Documentation(info="<html>
<p>
Controller for terminal VAV box with hot water reheat and pressure independent damper. 
It was implemented according to
<a href=\"https://newbuildings.org/sites/default/files/A-11_LG_VAV_Guide_3.6.2.pdf\">
[Advanced Variabled Air Volume System Design Guide]</a>, single maximum VAV reheat box
control.
The damper control signal <code>yDam</code> corresponds to the discharge air flow rate 
set-point, normalized to the nominal value.
</p>
<ul>
<li>
In cooling demand, the damper control signal <code>yDam</code> is modulated between 
a minimum value <code>ratVFloMin</code> (typically between 30% and 50%) and 1 
(corresponding to the nominal value).
The control signal for the reheat coil valve <code>yVal</code> is 0
(corresponding to the valve fully closed).
</li>
<li>
In heating demand, the damper control signal <code>yDam</code> is fixed at the minimum value 
<code>ratVFloMin</code>. 
The control signal for the reheat coil valve <code>yVal</code> is modulated between
0 and 1 (corresponding to the valve fully open).
</li>
</ul>
<p align=\"center\">
<img alt=\"image\" src=\"modelica://Buildings/Resources/Images/Examples/VAVReheat/vavBoxSingleMax.png\" border=\"1\"/>
</p>
<br/>
</html>",       revisions="<html>
<ul>
<li>
April 24, 2020, by Jianjun Hu:<br/>
Refactored the model to implement a single maximum control logic.
The previous implementation led to a maximum air flow rate in heating demand.<br/>
The input connector <code>TDis</code> is removed. This is non backward compatible.<br/>
This is for 
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/1873\">issue 1873</a>.
</li>
<li>
September 20, 2017, by Michael Wetter:<br/>
Removed blocks with blocks from CDL package.
</li>
</ul>
</html>"),Diagram(coordinateSystem(extent={{-100,-100},{100,100}})));
      end RoomVAV;

      model State
        "Block that outputs the mode if the state is active, or zero otherwise"
        extends Modelica.StateGraph.StepWithSignal;
       parameter OperationModes mode "Enter enumeration of mode";
        Modelica.Blocks.Interfaces.IntegerOutput y "Mode signal (=0 if not active)"
          annotation (Placement(transformation(extent={{100,-70},{120,-50}})));
      equation
         y = if localActive then Integer(mode) else 0;
        annotation (Icon(graphics={Text(
                extent={{-82,96},{82,-84}},
                lineColor={0,0,255},
                textString="state")}));
      end State;

      block ControllerOve
        "Multizone AHU controller that composes subsequences for controlling fan speed, dampers, and supply air temperature"

        parameter Real samplePeriod(
          final unit="s",
          final quantity="Time")=120
          "Sample period of component, set to the same value to the trim and respond sequence";

        parameter Boolean have_perZonRehBox=true
          "Check if there is any VAV-reheat boxes on perimeter zones"
          annotation (Dialog(group="System and building parameters"));

        parameter Boolean have_duaDucBox=false
          "Check if the AHU serves dual duct boxes"
          annotation (Dialog(group="System and building parameters"));

        parameter Boolean have_airFloMeaSta=false
          "Check if the AHU has AFMS (Airflow measurement station)"
          annotation (Dialog(group="System and building parameters"));

        // ----------- Parameters for economizer control -----------
        parameter Boolean use_enthalpy=false
          "Set to true if enthalpy measurement is used in addition to temperature measurement"
          annotation (Dialog(tab="Economizer"));

        parameter Real delta(
          final unit="s",
          final quantity="Time")=5
          "Time horizon over which the outdoor air flow measurment is averaged"
          annotation (Dialog(tab="Economizer"));

        parameter Real delTOutHis(
          final unit="K",
          final displayUnit="K",
          final quantity="TemperatureDifference")=1
          "Delta between the temperature hysteresis high and low limit"
          annotation (Dialog(tab="Economizer"));

        parameter Real delEntHis(
          final unit="J/kg",
          final quantity="SpecificEnergy")=1000
          "Delta between the enthalpy hysteresis high and low limits"
          annotation (Dialog(tab="Economizer", enable=use_enthalpy));

        parameter Real retDamPhyPosMax(
          final min=0,
          final max=1,
          final unit="1") = 1
          "Physically fixed maximum position of the return air damper"
          annotation (Dialog(tab="Economizer", group="Damper limits"));

        parameter Real retDamPhyPosMin(
          final min=0,
          final max=1,
          final unit="1") = 0
          "Physically fixed minimum position of the return air damper"
          annotation (Dialog(tab="Economizer", group="Damper limits"));

        parameter Real outDamPhyPosMax(
          final min=0,
          final max=1,
          final unit="1") = 1
          "Physically fixed maximum position of the outdoor air damper"
          annotation (Dialog(tab="Economizer", group="Damper limits"));

        parameter Real outDamPhyPosMin(
          final min=0,
          final max=1,
          final unit="1") = 0
          "Physically fixed minimum position of the outdoor air damper"
          annotation (Dialog(tab="Economizer", group="Damper limits"));

        parameter Buildings.Controls.OBC.CDL.Types.SimpleController controllerTypeMinOut=
          Buildings.Controls.OBC.CDL.Types.SimpleController.PI
          "Type of controller"
          annotation (Dialog(group="Economizer PID controller"));

        parameter Real kMinOut(final unit="1")=0.05
          "Gain of controller for minimum outdoor air intake"
          annotation (Dialog(group="Economizer PID controller"));

        parameter Real TiMinOut(
          final unit="s",
          final quantity="Time")=1200
          "Time constant of controller for minimum outdoor air intake"
          annotation (Dialog(group="Economizer PID controller",
            enable=controllerTypeMinOut == Buildings.Controls.OBC.CDL.Types.SimpleController.PI
                or controllerTypeMinOut == Buildings.Controls.OBC.CDL.Types.SimpleController.PID));

        parameter Real TdMinOut(
          final unit="s",
          final quantity="Time")=0.1
          "Time constant of derivative block for minimum outdoor air intake"
          annotation (Dialog(group="Economizer PID controller",
            enable=controllerTypeMinOut == Buildings.Controls.OBC.CDL.Types.SimpleController.PD
                or controllerTypeMinOut == Buildings.Controls.OBC.CDL.Types.SimpleController.PID));

        parameter Boolean use_TMix=true
          "Set to true if mixed air temperature measurement is enabled"
           annotation(Dialog(group="Economizer freeze protection"));

        parameter Boolean use_G36FrePro=false
          "Set to true to use G36 freeze protection"
          annotation(Dialog(group="Economizer freeze protection"));

        parameter Buildings.Controls.OBC.CDL.Types.SimpleController controllerTypeFre=
          Buildings.Controls.OBC.CDL.Types.SimpleController.PI
          "Type of controller"
          annotation(Dialog(group="Economizer freeze protection", enable=use_TMix));

        parameter Real kFre(final unit="1/K") = 0.1
          "Gain for mixed air temperature tracking for freeze protection, used if use_TMix=true"
           annotation(Dialog(group="Economizer freeze protection", enable=use_TMix));

        parameter Real TiFre(
          final unit="s",
          final quantity="Time",
          final max=TiMinOut)=120
          "Time constant of controller for mixed air temperature tracking for freeze protection. Require TiFre < TiMinOut"
           annotation(Dialog(group="Economizer freeze protection",
             enable=use_TMix
               and (controllerTypeFre == Buildings.Controls.OBC.CDL.Types.SimpleController.PI
                 or controllerTypeFre == Buildings.Controls.OBC.CDL.Types.SimpleController.PID)));

        parameter Real TdFre(
          final unit="s",
          final quantity="Time")=0.1
          "Time constant of derivative block for freeze protection"
          annotation (Dialog(group="Economizer freeze protection",
            enable=use_TMix and
                (controllerTypeFre == Buildings.Controls.OBC.CDL.Types.SimpleController.PD
                or controllerTypeFre == Buildings.Controls.OBC.CDL.Types.SimpleController.PID)));

        parameter Real TFreSet(
           final unit="K",
           final displayUnit="degC",
           final quantity="ThermodynamicTemperature")= 279.15
          "Lower limit for mixed air temperature for freeze protection, used if use_TMix=true"
           annotation(Dialog(group="Economizer freeze protection", enable=use_TMix));

        parameter Real yMinDamLim=0
          "Lower limit of damper position limits control signal output"
          annotation (Dialog(tab="Economizer", group="Damper limits"));

        parameter Real yMaxDamLim=1
          "Upper limit of damper position limits control signal output"
          annotation (Dialog(tab="Economizer", group="Damper limits"));

        parameter Real retDamFulOpeTim(
          final unit="s",
          final quantity="Time")=180
          "Time period to keep RA damper fully open before releasing it for minimum outdoor airflow control
    at disable to avoid pressure fluctuations"
          annotation (Dialog(tab="Economizer", group="Economizer delays at disable"));

        parameter Real disDel(
          final unit="s",
          final quantity="Time")=15
          "Short time delay before closing the OA damper at disable to avoid pressure fluctuations"
          annotation (Dialog(tab="Economizer", group="Economizer delays at disable"));

        // ----------- parameters for fan speed control  -----------
        parameter Real pIniSet(
          final unit="Pa",
          final displayUnit="Pa",
          final quantity="PressureDifference")=60
          "Initial pressure setpoint for fan speed control"
          annotation (Dialog(tab="Fan speed", group="Trim and respond for reseting duct static pressure setpoint"));

        parameter Real pMinSet(
          final unit="Pa",
          final displayUnit="Pa",
          final quantity="PressureDifference")=25
          "Minimum pressure setpoint for fan speed control"
          annotation (Dialog(tab="Fan speed", group="Trim and respond for reseting duct static pressure setpoint"));

        parameter Real pMaxSet(
          final unit="Pa",
          final displayUnit="Pa",
          final quantity="PressureDifference")=400
          "Maximum pressure setpoint for fan speed control"
          annotation (Dialog(tab="Fan speed", group="Trim and respond for reseting duct static pressure setpoint"));

        parameter Real pDelTim(
          final unit="s",
          final quantity="Time")=600
          "Delay time after which trim and respond is activated"
          annotation (Dialog(tab="Fan speed", group="Trim and respond for reseting duct static pressure setpoint"));

        parameter Integer pNumIgnReq=2
          "Number of ignored requests for fan speed control"
          annotation (Dialog(tab="Fan speed", group="Trim and respond for reseting duct static pressure setpoint"));

        parameter Real pTriAmo(
          final unit="Pa",
          final displayUnit="Pa",
          final quantity="PressureDifference")=-12.0
          "Trim amount for fan speed control"
          annotation (Dialog(tab="Fan speed", group="Trim and respond for reseting duct static pressure setpoint"));

        parameter Real pResAmo(
          final unit="Pa",
          final displayUnit="Pa",
          final quantity="PressureDifference")=15
          "Respond amount (must be opposite in to triAmo) for fan speed control"
          annotation (Dialog(tab="Fan speed", group="Trim and respond for reseting duct static pressure setpoint"));

        parameter Real pMaxRes(
          final unit="Pa",
          final displayUnit="Pa",
          final quantity="PressureDifference")=32
          "Maximum response per time interval (same sign as resAmo) for fan speed control"
          annotation (Dialog(tab="Fan speed", group="Trim and respond for reseting duct static pressure setpoint"));

        parameter Buildings.Controls.OBC.CDL.Types.SimpleController
          controllerTypeFanSpe=Buildings.Controls.OBC.CDL.Types.SimpleController.PI "Type of controller"
          annotation (Dialog(group="Fan speed PID controller"));

        parameter Real kFanSpe(final unit="1")=0.1
          "Gain of fan fan speed controller, normalized using pMaxSet"
          annotation (Dialog(group="Fan speed PID controller"));

        parameter Real TiFanSpe(
          final unit="s",
          final quantity="Time")=60
          "Time constant of integrator block for fan speed"
          annotation (Dialog(group="Fan speed PID controller",
            enable=controllerTypeFanSpe == Buildings.Controls.OBC.CDL.Types.SimpleController.PI
                or controllerTypeFanSpe == Buildings.Controls.OBC.CDL.Types.SimpleController.PID));

        parameter Real TdFanSpe(
          final unit="s",
          final quantity="Time")=0.1
          "Time constant of derivative block for fan speed"
          annotation (Dialog(group="Fan speed PID controller",
            enable=controllerTypeFanSpe == Buildings.Controls.OBC.CDL.Types.SimpleController.PD
                or controllerTypeFanSpe == Buildings.Controls.OBC.CDL.Types.SimpleController.PID));

        parameter Real yFanMax=1 "Maximum allowed fan speed"
          annotation (Dialog(group="Fan speed PID controller"));

        parameter Real yFanMin=0.1 "Lowest allowed fan speed if fan is on"
          annotation (Dialog(group="Fan speed PID controller"));

        // ----------- parameters for minimum outdoor airflow setting  -----------
        parameter Real VPriSysMax_flow(
          final unit="m3/s",
          final quantity="VolumeFlowRate")
          "Maximum expected system primary airflow at design stage"
          annotation (Dialog(tab="Minimum outdoor airflow rate", group="Nominal conditions"));

        parameter Real peaSysPop "Peak system population"
          annotation (Dialog(tab="Minimum outdoor airflow rate", group="Nominal conditions"));

        // ----------- parameters for supply air temperature control  -----------
        parameter Real TSupSetMin(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")=285.15
          "Lowest cooling supply air temperature setpoint"
          annotation (Dialog(tab="Supply air temperature", group="Temperature limits"));

        parameter Real TSupSetMax(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")=291.15
          "Highest cooling supply air temperature setpoint. It is typically 18 degC (65 degF) in mild and dry climates, 16 degC (60 degF) or lower in humid climates"
          annotation (Dialog(tab="Supply air temperature", group="Temperature limits"));

        parameter Real TSupSetDes(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")=286.15
          "Nominal supply air temperature setpoint"
          annotation (Dialog(tab="Supply air temperature", group="Temperature limits"));

        parameter Real TOutMin(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")=289.15
          "Lower value of the outdoor air temperature reset range. Typically value is 16 degC (60 degF)"
          annotation (Dialog(tab="Supply air temperature", group="Temperature limits"));

        parameter Real TOutMax(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")=294.15
          "Higher value of the outdoor air temperature reset range. Typically value is 21 degC (70 degF)"
          annotation (Dialog(tab="Supply air temperature", group="Temperature limits"));

        parameter Real iniSetSupTem(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")=supTemSetPoi.maxSet
          "Initial setpoint for supply temperature control"
          annotation (Dialog(tab="Supply air temperature", group="Trim and respond for reseting TSup setpoint"));

        parameter Real maxSetSupTem(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")=supTemSetPoi.TSupSetMax
          "Maximum setpoint for supply temperature control"
          annotation (Dialog(tab="Supply air temperature", group="Trim and respond for reseting TSup setpoint"));

        parameter Real minSetSupTem(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")=supTemSetPoi.TSupSetDes
          "Minimum setpoint for supply temperature control"
          annotation (Dialog(tab="Supply air temperature", group="Trim and respond for reseting TSup setpoint"));

        parameter Real delTimSupTem(
          final unit="s",
          final quantity="Time")=600
          "Delay timer for supply temperature control"
          annotation (Dialog(tab="Supply air temperature", group="Trim and respond for reseting TSup setpoint"));

        parameter Integer numIgnReqSupTem=2
          "Number of ignorable requests for supply temperature control"
          annotation (Dialog(tab="Supply air temperature", group="Trim and respond for reseting TSup setpoint"));

        parameter Real triAmoSupTem(
          final unit="K",
          final displayUnit="K",
          final quantity="TemperatureDifference")=0.1
          "Trim amount for supply temperature control"
          annotation (Dialog(tab="Supply air temperature", group="Trim and respond for reseting TSup setpoint"));

        parameter Real resAmoSupTem(
          final unit="K",
          final displayUnit="K",
          final quantity="TemperatureDifference")=-0.2
          "Response amount for supply temperature control"
          annotation (Dialog(tab="Supply air temperature", group="Trim and respond for reseting TSup setpoint"));

        parameter Real maxResSupTem(
          final unit="K",
          final displayUnit="K",
          final quantity="TemperatureDifference")=-0.6
          "Maximum response per time interval for supply temperature control"
          annotation (Dialog(tab="Supply air temperature", group="Trim and respond for reseting TSup setpoint"));

        parameter Buildings.Controls.OBC.CDL.Types.SimpleController controllerTypeTSup=
            Buildings.Controls.OBC.CDL.Types.SimpleController.PI
          "Type of controller for supply air temperature signal"
          annotation (Dialog(group="Supply air temperature"));

        parameter Real kTSup(final unit="1/K")=0.05
          "Gain of controller for supply air temperature signal"
          annotation (Dialog(group="Supply air temperature"));

        parameter Real TiTSup(
          final unit="s",
          final quantity="Time")=600
          "Time constant of integrator block for supply air temperature control signal"
          annotation (Dialog(group="Supply air temperature",
            enable=controllerTypeTSup == Buildings.Controls.OBC.CDL.Types.SimpleController.PI
                or controllerTypeTSup == Buildings.Controls.OBC.CDL.Types.SimpleController.PID));

        parameter Real TdTSup(
          final unit="s",
          final quantity="Time")=0.1
          "Time constant of integrator block for supply air temperature control signal"
          annotation (Dialog(group="Supply air temperature",
            enable=controllerTypeTSup == Buildings.Controls.OBC.CDL.Types.SimpleController.PD
                or controllerTypeTSup == Buildings.Controls.OBC.CDL.Types.SimpleController.PID));

        parameter Real uHeaMax(min=-0.9)=-0.25
          "Upper limit of controller signal when heating coil is off. Require -1 < uHeaMax < uCooMin < 1."
          annotation (Dialog(group="Supply air temperature"));

        parameter Real uCooMin(max=0.9)=0.25
          "Lower limit of controller signal when cooling coil is off. Require -1 < uHeaMax < uCooMin < 1."
          annotation (Dialog(group="Supply air temperature"));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput TZonHeaSet(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")
          "Zone air temperature heating setpoint"
          annotation (Placement(transformation(extent={{-240,280},{-200,320}}),
              iconTransformation(extent={{-240,320},{-200,360}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput TZonCooSet(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")
          "Zone air temperature cooling setpoint"
          annotation (Placement(transformation(extent={{-240,250},{-200,290}}),
              iconTransformation(extent={{-240,290},{-200,330}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput ducStaPre(
          final unit="Pa",
          final displayUnit="Pa")
          "Measured duct static pressure"
          annotation (Placement(transformation(extent={{-240,190},{-200,230}}),
              iconTransformation(extent={{-240,230},{-200,270}})));
        Buildings.Controls.OBC.CDL.Interfaces.RealInput TOut(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") "Outdoor air temperature"
          annotation (Placement(transformation(extent={{-240,220},{-200,260}}),
              iconTransformation(extent={{-240,260},{-200,300}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput sumDesZonPop(
          final min=0,
          final unit="1")
          "Sum of the design population of the zones in the group"
          annotation (Placement(transformation(extent={{-240,160},{-200,200}}),
              iconTransformation(extent={{-240,170},{-200,210}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput VSumDesPopBreZon_flow(
          final min=0,
          final unit = "m3/s",
          final quantity = "VolumeFlowRate")
          "Sum of the population component design breathing zone flow rate"
          annotation (Placement(transformation(extent={{-240,130},{-200,170}}),
              iconTransformation(extent={{-240,140},{-200,180}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput VSumDesAreBreZon_flow(
          final min=0,
          final unit = "m3/s",
          final quantity = "VolumeFlowRate")
          "Sum of the area component design breathing zone flow rate"
          annotation (Placement(transformation(extent={{-240,100},{-200,140}}),
              iconTransformation(extent={{-240,110},{-200,150}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput uDesSysVenEff(
          final min=0,
          final unit = "1")
          "Design system ventilation efficiency, equals to the minimum of all zones ventilation efficiency"
          annotation (Placement(transformation(extent={{-240,70},{-200,110}}),
              iconTransformation(extent={{-240,80},{-200,120}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput VSumUncOutAir_flow(
          final min=0,
          final unit = "m3/s",
          final quantity = "VolumeFlowRate")
          "Sum of all zones required uncorrected outdoor airflow rate"
          annotation (Placement(transformation(extent={{-240,40},{-200,80}}),
              iconTransformation(extent={{-240,50},{-200,90}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput VSumSysPriAir_flow(
          final min=0,
          final unit = "m3/s",
          final quantity = "VolumeFlowRate")
          "System primary airflow rate, equals to the sum of the measured discharged flow rate of all terminal units"
          annotation (Placement(transformation(extent={{-240,10},{-200,50}}),
              iconTransformation(extent={{-240,20},{-200,60}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput uOutAirFra_max(
          final min=0,
          final unit = "1")
          "Maximum zone outdoor air fraction, equals to the maximum of primary outdoor air fraction of all zones"
          annotation (Placement(transformation(extent={{-240,-20},{-200,20}}),
              iconTransformation(extent={{-240,-10},{-200,30}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput TSup(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")
          "Measured supply air temperature"
          annotation (Placement(transformation(extent={{-240,-50},{-200,-10}}),
              iconTransformation(extent={{-240,-70},{-200,-30}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput TOutCut(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")
          "OA temperature high limit cutoff. For differential dry bulb temeprature condition use return air temperature measurement"
          annotation (Placement(transformation(extent={{-240,-80},{-200,-40}}),
              iconTransformation(extent={{-240,-100},{-200,-60}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput hOut(
          final unit="J/kg",
          final quantity="SpecificEnergy") if use_enthalpy "Outdoor air enthalpy"
          annotation (Placement(transformation(extent={{-240,-110},{-200,-70}}),
              iconTransformation(extent={{-240,-130},{-200,-90}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput hOutCut(
          final unit="J/kg",
          final quantity="SpecificEnergy") if use_enthalpy
          "OA enthalpy high limit cutoff. For differential enthalpy use return air enthalpy measurement"
          annotation (Placement(transformation(extent={{-240,-140},{-200,-100}}),
              iconTransformation(extent={{-240,-160},{-200,-120}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput VOut_flow(
          final min=0,
          final unit = "m3/s",
          final quantity = "VolumeFlowRate")
          "Measured outdoor volumetric airflow rate"
          annotation (Placement(transformation(extent={{-240,-170},{-200,-130}}),
              iconTransformation(extent={{-240,-190},{-200,-150}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput TMix(
          final unit="K",
          final displayUnit="degC",
          final quantity = "ThermodynamicTemperature") if use_TMix
          "Measured mixed air temperature, used for freeze protection if use_TMix=true"
          annotation (Placement(transformation(extent={{-240,-200},{-200,-160}}),
              iconTransformation(extent={{-240,-230},{-200,-190}})));

        Buildings.Controls.OBC.CDL.Interfaces.IntegerInput uOpeMod
          "AHU operation mode status signal"
          annotation (Placement(transformation(extent={{-240,-230},{-200,-190}}),
              iconTransformation(extent={{-240,-270},{-200,-230}})));

        Buildings.Controls.OBC.CDL.Interfaces.IntegerInput uZonTemResReq
          "Zone cooling supply air temperature reset request"
          annotation (Placement(transformation(extent={{-240,-260},{-200,-220}}),
              iconTransformation(extent={{-240,-300},{-200,-260}})));

        Buildings.Controls.OBC.CDL.Interfaces.IntegerInput uZonPreResReq
          "Zone static pressure reset requests"
          annotation (Placement(transformation(extent={{-240,-290},{-200,-250}}),
              iconTransformation(extent={{-240,-330},{-200,-290}})));

        Buildings.Controls.OBC.CDL.Interfaces.IntegerInput uFreProSta if
            use_G36FrePro
         "Freeze protection status, used if use_G36FrePro=true"
          annotation (Placement(transformation(extent={{-240,-320},{-200,-280}}),
              iconTransformation(extent={{-240,-360},{-200,-320}})));

        Buildings.Controls.OBC.CDL.Interfaces.BooleanOutput ySupFan
          "Supply fan status, true if fan should be on"
          annotation (Placement(transformation(extent={{200,260},{240,300}}),
              iconTransformation(extent={{200,280},{240,320}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealOutput ySupFanSpe(
          final min=0,
          final max=1,
          final unit="1") "Supply fan speed"
          annotation (Placement(transformation(extent={{200,190},{240,230}}),
              iconTransformation(extent={{200,220},{240,260}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealOutput TSupSet(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")
          "Setpoint for supply air temperature"
          annotation (Placement(transformation(extent={{200,160},{240,200}}),
              iconTransformation(extent={{200,160},{240,200}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealOutput VDesUncOutAir_flow(
          final min=0,
          final unit = "m3/s",
          final quantity = "VolumeFlowRate")
          "Design uncorrected minimum outdoor airflow rate"
          annotation (Placement(transformation(extent={{200,120},{240,160}}),
            iconTransformation(extent={{200,100},{240,140}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealOutput yAveOutAirFraPlu(
          final min=0,
          final unit = "1")
          "Average outdoor air flow fraction plus 1"
          annotation (Placement(transformation(extent={{200,80},{240,120}}),
            iconTransformation(extent={{200,40},{240,80}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealOutput VEffOutAir_flow(
          final min=0,
          final unit = "m3/s",
          final quantity = "VolumeFlowRate")
          "Effective minimum outdoor airflow setpoint"
          annotation (Placement(transformation(extent={{200,40},{240,80}}),
            iconTransformation(extent={{200,-20},{240,20}})));

        Buildings.Controls.OBC.CDL.Interfaces.BooleanOutput yReqOutAir
          "True if the AHU supply fan is on and the zone is in occupied mode"
          annotation (Placement(transformation(extent={{200,0},{240,40}}),
              iconTransformation(extent={{200,-80},{240,-40}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealOutput yHea(
          final min=0,
          final max=1,
          final unit="1")
          "Control signal for heating"
          annotation (Placement(transformation(extent={{200,-50},{240,-10}}),
              iconTransformation(extent={{200,-140},{240,-100}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealOutput yCoo(
          final min=0,
          final max=1,
          final unit="1") "Control signal for cooling"
          annotation (Placement(transformation(extent={{200,-110},{240,-70}}),
              iconTransformation(extent={{200,-200},{240,-160}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealOutput yRetDamPos(
          final min=0,
          final max=1,
          final unit="1") "Return air damper position"
          annotation (Placement(transformation(extent={{200,-170},{240,-130}}),
              iconTransformation(extent={{200,-260},{240,-220}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealOutput yOutDamPos(
          final min=0,
          final max=1,
          final unit="1") "Outdoor air damper position"
          annotation (Placement(transformation(extent={{200,-210},{240,-170}}),
              iconTransformation(extent={{200,-320},{240,-280}})));

        Buildings.Controls.OBC.CDL.Continuous.Average TZonSetPoiAve
          "Average of all zone set points"
          annotation (Placement(transformation(extent={{-160,270},{-140,290}})));

        FiveZoneAir.VAVReheat.Controls.SupplyFanOve supFan(
          final samplePeriod=samplePeriod,
          final have_perZonRehBox=have_perZonRehBox,
          final have_duaDucBox=have_duaDucBox,
          final have_airFloMeaSta=have_airFloMeaSta,
          final iniSet=pIniSet,
          final minSet=pMinSet,
          final maxSet=pMaxSet,
          final delTim=pDelTim,
          final numIgnReq=pNumIgnReq,
          final triAmo=pTriAmo,
          final resAmo=pResAmo,
          final maxRes=pMaxRes,
          final controllerType=controllerTypeFanSpe,
          final k=kFanSpe,
          final Ti=TiFanSpe,
          final Td=TdFanSpe,
          final yFanMax=yFanMax,
          final yFanMin=yFanMin)
          "Supply fan controller"
          annotation (Placement(transformation(extent={{-160,200},{-140,220}})));

        FiveZoneAir.VAVReheat.Controls.SupplyTemperatureOve supTemSetPoi(
          final samplePeriod=samplePeriod,
          final TSupSetMin=TSupSetMin,
          final TSupSetMax=TSupSetMax,
          final TSupSetDes=TSupSetDes,
          final TOutMin=TOutMin,
          final TOutMax=TOutMax,
          final iniSet=iniSetSupTem,
          final maxSet=maxSetSupTem,
          final minSet=minSetSupTem,
          final delTim=delTimSupTem,
          final numIgnReq=numIgnReqSupTem,
          final triAmo=triAmoSupTem,
          final resAmo=resAmoSupTem,
          final maxRes=maxResSupTem) "Setpoint for supply temperature"
          annotation (Placement(transformation(extent={{0,170},{20,190}})));

        Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.OutdoorAirFlow.AHU
          sysOutAirSet(final VPriSysMax_flow=VPriSysMax_flow, final peaSysPop=
              peaSysPop) "Minimum outdoor airflow setpoint"
          annotation (Placement(transformation(extent={{-40,70},{-20,90}})));

        Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.Economizers.Controller eco(
          final use_enthalpy=use_enthalpy,
          final delTOutHis=delTOutHis,
          final delEntHis=delEntHis,
          final retDamFulOpeTim=retDamFulOpeTim,
          final disDel=disDel,
          final controllerTypeMinOut=controllerTypeMinOut,
          final kMinOut=kMinOut,
          final TiMinOut=TiMinOut,
          final TdMinOut=TdMinOut,
          final retDamPhyPosMax=retDamPhyPosMax,
          final retDamPhyPosMin=retDamPhyPosMin,
          final outDamPhyPosMax=outDamPhyPosMax,
          final outDamPhyPosMin=outDamPhyPosMin,
          final uHeaMax=uHeaMax,
          final uCooMin=uCooMin,
          final uOutDamMax=(uHeaMax + uCooMin)/2,
          final uRetDamMin=(uHeaMax + uCooMin)/2,
          final TFreSet=TFreSet,
          final controllerTypeFre=controllerTypeFre,
          final kFre=kFre,
          final TiFre=TiFre,
          final TdFre=TdFre,
          final delta=delta,
          final use_TMix=use_TMix,
          final use_G36FrePro=use_G36FrePro) "Economizer controller"
          annotation (Placement(transformation(extent={{140,-170},{160,-150}})));

        Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.SupplySignals val(
          final controllerType=controllerTypeTSup,
          final kTSup=kTSup,
          final TiTSup=TiTSup,
          final TdTSup=TdTSup,
          final uHeaMax=uHeaMax,
          final uCooMin=uCooMin) "AHU coil valve control"
          annotation (Placement(transformation(extent={{80,-70},{100,-50}})));

      protected
        Buildings.Controls.OBC.CDL.Continuous.Division VOut_flow_normalized(
          u1(final unit="m3/s"),
          u2(final unit="m3/s"),
          y(final unit="1"))
          "Normalization of outdoor air flow intake by design minimum outdoor air intake"
          annotation (Placement(transformation(extent={{20,-130},{40,-110}})));

      equation
        connect(eco.yRetDamPos, yRetDamPos)
          annotation (Line(points={{161.25,-157.5},{180,-157.5},{180,-150},{220,-150}},
            color={0,0,127}));
        connect(eco.yOutDamPos, yOutDamPos)
          annotation (Line(points={{161.25,-162.5},{180,-162.5},{180,-190},{220,-190}},
            color={0,0,127}));
        connect(eco.uSupFan, supFan.ySupFan)
          annotation (Line(points={{138.75,-165},{-84,-165},{-84,217},{-138,217}},
            color={255,0,255}));
        connect(supFan.ySupFanSpe, ySupFanSpe)
          annotation (Line(points={{-138,210},{220,210}},
            color={0,0,127}));
        connect(TOut, eco.TOut)
          annotation (Line(points={{-220,240},{-60,240},{-60,-150.625},{138.75,-150.625}},
            color={0,0,127}));
        connect(eco.TOutCut, TOutCut)
          annotation (Line(points={{138.75,-152.5},{-74,-152.5},{-74,-60},{-220,-60}},
            color={0,0,127}));
        connect(eco.hOut, hOut)
          annotation (Line(points={{138.75,-154.375},{-78,-154.375},{-78,-90},{-220,-90}},
            color={0,0,127}));
        connect(eco.hOutCut, hOutCut)
          annotation (Line(points={{138.75,-155.625},{-94,-155.625},{-94,-120},{-220,-120}},
            color={0,0,127}));
        connect(eco.uOpeMod, uOpeMod)
          annotation (Line(points={{138.75,-166.875},{60,-166.875},{60,-210},{-220,-210}},
            color={255,127,0}));
        connect(supTemSetPoi.TSupSet, TSupSet)
          annotation (Line(points={{22,180},{220,180}}, color={0,0,127}));
        connect(supTemSetPoi.TOut, TOut)
          annotation (Line(points={{-2,184},{-60,184},{-60,240},{-220,240}},
            color={0,0,127}));
        connect(supTemSetPoi.uSupFan, supFan.ySupFan)
          annotation (Line(points={{-2,176},{-84,176},{-84,217},{-138,217}},
            color={255,0,255}));
        connect(supTemSetPoi.uZonTemResReq, uZonTemResReq)
          annotation (Line(points={{-2,180},{-52,180},{-52,-240},{-220,-240}},
            color={255,127,0}));
        connect(supTemSetPoi.uOpeMod, uOpeMod)
          annotation (Line(points={{-2,172},{-48,172},{-48,-210},{-220,-210}},
            color={255,127,0}));
        connect(supFan.uOpeMod, uOpeMod)
          annotation (Line(points={{-162,218},{-180,218},{-180,-210},{-220,-210}},
            color={255,127,0}));
        connect(supFan.uZonPreResReq, uZonPreResReq)
          annotation (Line(points={{-162,207},{-176,207},{-176,-270},{-220,-270}},
            color={255,127,0}));
        connect(supFan.ducStaPre, ducStaPre)
          annotation (Line(points={{-162,202},{-192,202},{-192,210},{-220,210}},
            color={0,0,127}));
        connect(supTemSetPoi.TZonSetAve, TZonSetPoiAve.y)
          annotation (Line(points={{-2,188},{-20,188},{-20,280},{-138,280}},
            color={0,0,127}));
        connect(supFan.ySupFan, ySupFan)
          annotation (Line(points={{-138,217},{60,217},{60,280},{220,280}},
            color={255,0,255}));
        connect(TZonSetPoiAve.u2, TZonCooSet)
          annotation (Line(points={{-162,274},{-180,274},{-180,270},{-220,270}},
            color={0,0,127}));
        connect(eco.TMix, TMix)
          annotation (Line(points={{138.75,-163.125},{-12,-163.125},{-12,-180},{-220,-180}},
            color={0,0,127}));
        connect(TSup, val.TSup)
          annotation (Line(points={{-220,-30},{-66,-30},{-66,-65},{78,-65}},
            color={0,0,127}));
        connect(supFan.ySupFan, val.uSupFan)
          annotation (Line(points={{-138,217},{-84,217},{-84,-55},{78,-55}},
            color={255,0,255}));
        connect(val.uTSup, eco.uTSup)
          annotation (Line(points={{102,-56},{120,-56},{120,-157.5},{138.75,-157.5}},
            color={0,0,127}));
        connect(val.yHea, yHea)
          annotation (Line(points={{102,-60},{180,-60},{180,-30},{220,-30}},
            color={0,0,127}));
        connect(val.yCoo, yCoo)
          annotation (Line(points={{102,-64},{180,-64},{180,-90},{220,-90}},
            color={0,0,127}));
        connect(supTemSetPoi.TSupSet, val.TSupSet)
          annotation (Line(points={{22,180},{60,180},{60,-60},{78,-60}},
            color={0,0,127}));
        connect(TZonHeaSet, TZonSetPoiAve.u1)
          annotation (Line(points={{-220,300},{-180,300},{-180,286},{-162,286}},
            color={0,0,127}));
        connect(eco.uFreProSta, uFreProSta)
          annotation (Line(points={{138.75,-169.375},{66,-169.375},{66,-300},{-220,-300}},
            color={255,127,0}));
        connect(eco.VOut_flow_normalized, VOut_flow_normalized.y)
          annotation (Line(points={{138.75,-159.375},{60,-159.375},{60,-120},{42,-120}},
            color={0,0,127}));
        connect(VOut_flow_normalized.u1, VOut_flow)
          annotation (Line(points={{18,-114},{-160,-114},{-160,-150},{-220,-150}},
            color={0,0,127}));
        connect(sysOutAirSet.VDesUncOutAir_flow, VDesUncOutAir_flow) annotation (Line(
              points={{-18,88},{0,88},{0,140},{220,140}}, color={0,0,127}));
        connect(sysOutAirSet.VDesOutAir_flow, VOut_flow_normalized.u2) annotation (
            Line(points={{-18,82},{0,82},{0,-126},{18,-126}}, color={0,0,127}));
        connect(sysOutAirSet.effOutAir_normalized, eco.VOutMinSet_flow_normalized)
          annotation (Line(points={{-18,75},{-4,75},{-4,-161.25},{138.75,-161.25}},
              color={0,0,127}));
        connect(supFan.ySupFan, sysOutAirSet.uSupFan) annotation (Line(points={{-138,217},
                {-84,217},{-84,73},{-42,73}}, color={255,0,255}));
        connect(uOpeMod, sysOutAirSet.uOpeMod) annotation (Line(points={{-220,-210},{-48,
                -210},{-48,71},{-42,71}}, color={255,127,0}));
        connect(sysOutAirSet.yAveOutAirFraPlu, yAveOutAirFraPlu) annotation (Line(
              points={{-18,85},{20,85},{20,100},{220,100}}, color={0,0,127}));
        connect(sysOutAirSet.VEffOutAir_flow, VEffOutAir_flow) annotation (Line(
              points={{-18,78},{20,78},{20,60},{220,60}}, color={0,0,127}));
        connect(sysOutAirSet.yReqOutAir, yReqOutAir) annotation (Line(points={{-18,
                72},{16,72},{16,20},{220,20}}, color={255,0,255}));
        connect(sysOutAirSet.sumDesZonPop, sumDesZonPop) annotation (Line(points={{-42,
                89},{-120,89},{-120,180},{-220,180}}, color={0,0,127}));
        connect(sysOutAirSet.VSumDesPopBreZon_flow, VSumDesPopBreZon_flow)
          annotation (Line(points={{-42,87},{-126,87},{-126,150},{-220,150}}, color={0,
                0,127}));
        connect(sysOutAirSet.VSumDesAreBreZon_flow, VSumDesAreBreZon_flow)
          annotation (Line(points={{-42,85},{-132,85},{-132,120},{-220,120}}, color={0,
                0,127}));
        connect(sysOutAirSet.uDesSysVenEff, uDesSysVenEff) annotation (Line(points={{-42,
                83},{-138,83},{-138,90},{-220,90}}, color={0,0,127}));
        connect(sysOutAirSet.VSumUncOutAir_flow, VSumUncOutAir_flow) annotation (Line(
              points={{-42,81},{-138,81},{-138,60},{-220,60}}, color={0,0,127}));
        connect(sysOutAirSet.VSumSysPriAir_flow, VSumSysPriAir_flow) annotation (Line(
              points={{-42,79},{-132,79},{-132,30},{-220,30}}, color={0,0,127}));
        connect(uOutAirFra_max, sysOutAirSet.uOutAirFra_max) annotation (Line(points={
                {-220,0},{-126,0},{-126,77},{-42,77}}, color={0,0,127}));

      annotation (defaultComponentName="conAHU",
          Diagram(coordinateSystem(extent={{-200,-320},{200,320}}, initialScale=0.2)),
          Icon(coordinateSystem(extent={{-200,-360},{200,360}}, initialScale=0.2),
              graphics={Rectangle(
                extent={{200,360},{-200,-360}},
                lineColor={0,0,0},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid), Text(
                extent={{-200,450},{200,372}},
                textString="%name",
                lineColor={0,0,255}),           Text(
                extent={{-200,348},{-116,332}},
                lineColor={0,0,0},
                textString="TZonHeaSet"),       Text(
                extent={{102,-48},{202,-68}},
                lineColor={255,0,255},
                textString="yReqOutAir"),       Text(
                extent={{-196,-238},{-122,-258}},
                lineColor={255,127,0},
                textString="uOpeMod"),          Text(
                extent={{-200,318},{-114,302}},
                lineColor={0,0,0},
                textString="TZonCooSet"),       Text(
                extent={{-198,260},{-120,242}},
                lineColor={0,0,0},
                textString="ducStaPre"),        Text(
                extent={{-198,288},{-162,272}},
                lineColor={0,0,0},
                textString="TOut"),             Text(
                extent={{-196,110},{-90,88}},
                lineColor={0,0,0},
                textString="uDesSysVenEff"),    Text(
                extent={{-196,140},{-22,118}},
                lineColor={0,0,0},
                textString="VSumDesAreBreZon_flow"),
                                                Text(
                extent={{-196,170},{-20,148}},
                lineColor={0,0,0},
                textString="VSumDesPopBreZon_flow"),
                                                Text(
                extent={{-196,200},{-88,182}},
                lineColor={0,0,0},
                textString="sumDesZonPop"),     Text(
                extent={{-200,-42},{-154,-62}},
                lineColor={0,0,0},
                textString="TSup"),             Text(
                extent={{-200,18},{-84,0}},
                lineColor={0,0,0},
                textString="uOutAirFra_max"),   Text(
                extent={{-196,48},{-62,30}},
                lineColor={0,0,0},
                textString="VSumSysPriAir_flow"),
                                                Text(
                extent={{-196,80},{-42,58}},
                lineColor={0,0,0},
                textString="VSumUncOutAir_flow"),
                                                Text(
                extent={{-200,-162},{-126,-180}},
                lineColor={0,0,0},
                textString="VOut_flow"),        Text(
                visible=use_enthalpy,
                extent={{-200,-130},{-134,-148}},
                lineColor={0,0,0},
                textString="hOutCut"),          Text(
                visible=use_enthalpy,
                extent={{-200,-100},{-160,-118}},
                lineColor={0,0,0},
                textString="hOut"),             Text(
                extent={{-198,-70},{-146,-86}},
                lineColor={0,0,0},
                textString="TOutCut"),          Text(
                visible=use_TMix,
                extent={{-200,-200},{-154,-218}},
                lineColor={0,0,0},
                textString="TMix"),             Text(
                extent={{-194,-270},{-68,-290}},
                lineColor={255,127,0},
                textString="uZonTemResReq"),    Text(
                extent={{-192,-300},{-74,-320}},
                lineColor={255,127,0},
                textString="uZonPreResReq"),    Text(
                visible=use_G36FrePro,
                extent={{-200,-330},{-110,-348}},
                lineColor={255,127,0},
                textString="uFreProSta"),       Text(
                extent={{106,252},{198,230}},
                lineColor={0,0,0},
                textString="ySupFanSpe"),       Text(
                extent={{122,192},{202,172}},
                lineColor={0,0,0},
                textString="TSupSet"),          Text(
                extent={{68,72},{196,52}},
                lineColor={0,0,0},
                textString="yAveOutAirFraPlu"), Text(
                extent={{48,132},{196,110}},
                lineColor={0,0,0},
                textString="VDesUncOutAir_flow"),
                                                Text(
                extent={{150,-104},{200,-126}},
                lineColor={0,0,0},
                textString="yHea"),             Text(
                extent={{94,-288},{200,-308}},
                lineColor={0,0,0},
                textString="yOutDamPos"),       Text(
                extent={{98,-228},{198,-248}},
                lineColor={0,0,0},
                textString="yRetDamPos"),       Text(
                extent={{78,14},{196,-6}},
                lineColor={0,0,0},
                textString="VEffOutAir_flow"),  Text(
                extent={{120,312},{202,292}},
                lineColor={255,0,255},
                textString="ySupFan"),          Text(
                extent={{150,-166},{200,-188}},
                lineColor={0,0,0},
                textString="yCoo")}),
      Documentation(info="<html>
<p>
Block that is applied for multizone VAV AHU control. It outputs the supply fan status
and the operation speed, outdoor and return air damper position, supply air
temperature setpoint and the valve position of the cooling and heating coils.
It is implemented according to the ASHRAE Guideline 36, PART 5.N.
</p>
<p>
The sequence consists of five subsequences.
</p>
<h4>Supply fan speed control</h4>
<p>
The fan speed control is implemented according to PART 5.N.1. It outputs
the boolean signal <code>ySupFan</code> to turn on or off the supply fan.
In addition, based on the pressure reset request <code>uZonPreResReq</code>
from the VAV zones controller, the
sequence resets the duct pressure setpoint, and uses this setpoint
to modulate the fan speed <code>ySupFanSpe</code> using a PI controller.
See
<a href=\"modelica://Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.SupplyFan\">
Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.SupplyFan</a>
for more detailed description.
</p>
<h4>Minimum outdoor airflow setting</h4>
<p>
According to current occupany, supply operation status <code>ySupFan</code>,
zone temperatures and the discharge air temperature, the sequence computes the
minimum outdoor airflow rate setpoint, which is used as input for the economizer control.
More detailed information can be found in
<a href=\"modelica://Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.OutdoorAirFlow\">
Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.OutdoorAirFlow</a>.
</p>
<h4>Economizer control</h4>
<p>
The block outputs outdoor and return air damper position, <code>yOutDamPos</code> and
<code>yRetDamPos</code>. First, it computes the position limits to satisfy the minimum
outdoor airflow requirement. Second, it determines the availability of the economizer based
on the outdoor condition. The dampers are modulated to track the supply air temperature
loop signal, which is calculated from the sequence below, subject to the minimum outdoor airflow
requirement and economizer availability. Optionally, there is also an override for freeze protection.
See
<a href=\"modelica://Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.Economizers.Controller\">
Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.Economizers.Controller</a>
for more detailed description.
</p>
<h4>Supply air temperature setpoint</h4>
<p>
Based on PART 5.N.2, the sequence first sets the maximum supply air temperature
based on reset requests collected from each zone <code>uZonTemResReq</code>. The
outdoor temperature <code>TOut</code> and operation mode <code>uOpeMod</code> are used
along with the maximum supply air temperature, for computing the supply air temperature
setpoint. See
<a href=\"modelica://Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.SupplyTemperature\">
Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.SupplyTemperature</a>
for more detailed description.
</p>
<h4>Coil valve control</h4>
<p>
The subsequence retrieves supply air temperature setpoint from previous sequence.
Along with the measured supply air temperature and the supply fan status, it
generates coil valve positions. See
<a href=\"modelica://Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.SupplySignals\">
Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.SupplySignals</a>
</p>
</html>",
      revisions="<html>
<ul>
<li>
March 16, 2020, by Jianjun Hu:<br/>
Reimplemented to add new block for specifying the minimum outdoor airfow setpoint.
This new block avoids vector-valued calculations.<br/>
This is for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/1829\">#1829</a>.
</li>
<li>
October 27, 2017, by Jianjun Hu:<br/>
First implementation.
</li>
</ul>
</html>"));
      end ControllerOve;

      block SupplyTemperatureOve
        "Supply air temperature setpoint for multi zone system"

        parameter Real TSupSetMin(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") = 285.15
          "Lowest cooling supply air temperature setpoint"
          annotation (Dialog(group="Temperatures"));
        parameter Real TSupSetMax(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") = 291.15
          "Highest cooling supply air temperature setpoint. It is typically 18 degC (65 degF) 
    in mild and dry climates, 16 degC (60 degF) or lower in humid climates"
          annotation (Dialog(group="Temperatures"));
        parameter Real TSupSetDes(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") = 286.15
          "Nominal supply air temperature setpoint"
          annotation (Dialog(group="Temperatures"));
        parameter Real TOutMin(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") = 289.15
          "Lower value of the outdoor air temperature reset range. Typically value is 16 degC (60 degF)"
          annotation (Dialog(group="Temperatures"));
        parameter Real TOutMax(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") = 294.15
          "Higher value of the outdoor air temperature reset range. Typically value is 21 degC (70 degF)"
          annotation (Dialog(group="Temperatures"));
        parameter Real TSupWarUpSetBac(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")=308.15
          "Supply temperature in warm up and set back mode"
          annotation (Dialog(group="Temperatures"));
        parameter Real iniSet(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") = maxSet
          "Initial setpoint"
          annotation (Dialog(group="Trim and respond logic"));
        parameter Real maxSet(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") = TSupSetMax
          "Maximum setpoint"
          annotation (Dialog(group="Trim and respond logic"));
        parameter Real minSet(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") = TSupSetDes
          "Minimum setpoint"
          annotation (Dialog(group="Trim and respond logic"));
        parameter Real delTim(
          final unit="s",
          final quantity="Time") = 600
          "Delay timer"
          annotation(Dialog(group="Trim and respond logic"));
        parameter Real samplePeriod(
          final unit="s",
          final quantity="Time",
          final min=1E-3) = 120
          "Sample period of component"
          annotation(Dialog(group="Trim and respond logic"));
        parameter Integer numIgnReq = 2
          "Number of ignorable requests for TrimResponse logic"
          annotation(Dialog(group="Trim and respond logic"));
        parameter Real triAmo(
          final unit="K",
          final displayUnit="K",
          final quantity="TemperatureDifference") = 0.1
          "Trim amount"
          annotation (Dialog(group="Trim and respond logic"));
        parameter Real resAmo(
          final unit="K",
          final displayUnit="K",
          final quantity="TemperatureDifference") = -0.2
          "Response amount"
          annotation (Dialog(group="Trim and respond logic"));
        parameter Real maxRes(
          final unit="K",
          final displayUnit="K",
          final quantity="TemperatureDifference") = -0.6
          "Maximum response per time interval"
          annotation (Dialog(group="Trim and respond logic"));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput TOut(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")
          "Outdoor air temperature"
          annotation (Placement(transformation(extent={{-180,40},{-140,80}}),
              iconTransformation(extent={{-140,20},{-100,60}})));
        Buildings.Controls.OBC.CDL.Interfaces.RealInput TZonSetAve(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")
          "Average of heating and cooling setpoint"
          annotation (Placement(transformation(extent={{-180,70},{-140,110}}),
              iconTransformation(extent={{-140,60},{-100,100}})));
        Buildings.Controls.OBC.CDL.Interfaces.BooleanInput uSupFan
          "Supply fan status"
          annotation (Placement(transformation(extent={{-180,-50},{-140,-10}}),
              iconTransformation(extent={{-140,-60},{-100,-20}})));
        Buildings.Controls.OBC.CDL.Interfaces.IntegerInput uOpeMod
          "System operation mode"
          annotation (Placement(transformation(extent={{-180,-120},{-140,-80}}),
              iconTransformation(extent={{-140,-100},{-100,-60}})));
        Buildings.Controls.OBC.CDL.Interfaces.IntegerInput uZonTemResReq
          "Zone cooling supply air temperature reset request"
          annotation (Placement( transformation(extent={{-180,0},{-140,40}}),
              iconTransformation(extent={{-140,-20},{-100,20}})));
        Buildings.Controls.OBC.CDL.Interfaces.RealOutput TSupSet(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")
          "Setpoint for supply air temperature"
          annotation (Placement(transformation(extent={{140,-20},{180,20}}),
              iconTransformation(extent={{100,-20},{140,20}})));

        Buildings.Controls.OBC.ASHRAE.G36_PR1.Generic.SetPoints.TrimAndRespond maxSupTemRes(
          final delTim=delTim,
          final iniSet=iniSet,
          final minSet=minSet,
          final maxSet=maxSet,
          final samplePeriod=samplePeriod,
          final numIgnReq=numIgnReq,
          final triAmo=triAmo,
          final resAmo=resAmo,
          final maxRes=maxRes) "Maximum cooling supply temperature reset"
          annotation (Placement(transformation(extent={{-100,20},{-80,40}})));

        Buildings.Utilities.IO.SignalExchange.Overwrite oveActTAirSup(
            description="Supply air temperature setpoint", u(
            unit="K",
            min=273.15 + 12,
            max=273.15 + 18)) "Overwrite the supply air temperature setpoint"
          annotation (Placement(transformation(extent={{80,40},{100,60}})));
      protected
        Buildings.Controls.OBC.CDL.Continuous.Line lin
          "Supply temperature distributes linearly between minimum and maximum supply 
    air temperature, according to outdoor temperature"
          annotation (Placement(transformation(extent={{20,40},{40,60}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant minOutTem(k=TOutMin)
          "Lower value of the outdoor air temperature reset range"
          annotation (Placement(transformation(extent={{-40,60},{-20,80}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant maxOutTem(k=TOutMax)
          "Higher value of the outdoor air temperature reset range"
          annotation (Placement(transformation(extent={{-40,20},{-20,40}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant minSupTem(k=TSupSetMin)
          "Lowest cooling supply air temperature setpoint"
          annotation (Placement(transformation(extent={{-100,-20},{-80,0}})));
        Buildings.Controls.OBC.CDL.Logical.And and2
          "Check if it is in Setup or Cool-down mode"
          annotation (Placement(transformation(extent={{-40,-60},{-20,-40}})));
        Buildings.Controls.OBC.CDL.Logical.And and1
          "Check if it is in Warmup or Setback mode"
          annotation (Placement(transformation(extent={{20,-100},{40,-80}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant supTemWarUpSetBac(k=
              TSupWarUpSetBac)
          "Supply temperature setpoint under warm-up and setback mode"
          annotation (Placement(transformation(extent={{20,-130},{40,-110}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi1
          "If operation mode is setup or cool-down, setpoint shall be 35 degC"
          annotation (Placement(transformation(extent={{80,-60},{100,-40}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi2
          "If operation mode is setup or cool-down, setpoint shall be TSupSetMin"
          annotation (Placement(transformation(extent={{20,-60},{40,-40}})));
        Buildings.Controls.OBC.CDL.Continuous.Limiter TDea(
          uMax=297.15,
          uMin=294.15)
          "Limiter that outputs the dead band value for the supply air temperature"
          annotation (Placement(transformation(extent={{-100,80},{-80,100}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi3
          "Check output regarding supply fan status"
          annotation (Placement(transformation(extent={{80,-10},{100,10}})));
        Buildings.Controls.OBC.CDL.Integers.LessThreshold intLesThr(
          threshold=Buildings.Controls.OBC.ASHRAE.G36_PR1.Types.OperationModes.warmUp)
          "Check if operation mode index is less than warm-up mode index (4)"
          annotation (Placement(transformation(extent={{-100,-60},{-80,-40}})));
        Buildings.Controls.OBC.CDL.Integers.GreaterThreshold intGreThr(
          threshold=Buildings.Controls.OBC.ASHRAE.G36_PR1.Types.OperationModes.occupied)
          "Check if operation mode index is greater than occupied mode index (1)"
          annotation (Placement(transformation(extent={{-100,-90},{-80,-70}})));
        Buildings.Controls.OBC.CDL.Integers.LessThreshold intLesThr1(
          threshold=Buildings.Controls.OBC.ASHRAE.G36_PR1.Types.OperationModes.unoccupied)
          "Check if operation mode index is less than unoccupied mode index (7)"
          annotation (Placement(transformation(extent={{-40,-100},{-20,-80}})));
        Buildings.Controls.OBC.CDL.Integers.GreaterThreshold intGreThr1(
          threshold=Buildings.Controls.OBC.ASHRAE.G36_PR1.Types.OperationModes.setUp)
          "Check if operation mode index is greater than set up mode index (3)"
          annotation (Placement(transformation(extent={{-40,-130},{-20,-110}})));

      equation
        connect(minOutTem.y, lin.x1)
          annotation (Line(points={{-18,70},{0,70},{0,58},{18,58}},
            color={0,0,127}));
        connect(TOut, lin.u)
          annotation (Line(points={{-160,60},{-100,60},{-100,50},{18,50}},
            color={0,0,127}));
        connect(maxOutTem.y, lin.x2)
          annotation (Line(points={{-18,30},{0,30},{0,46},{18,46}},
            color={0,0,127}));
        connect(minSupTem.y, lin.f2)
          annotation (Line(points={{-78,-10},{10,-10},{10,42},{18,42}},
            color={0,0,127}));
        connect(and1.y, swi1.u2)
          annotation (Line(points={{42,-90},{60,-90},{60,-50},{78,-50}},
            color={255,0,255}));
        connect(supTemWarUpSetBac.y, swi1.u1)
          annotation (Line(points={{42,-120},{68,-120},{68,-42},{78,-42}},
            color={0,0,127}));
        connect(and2.y, swi2.u2)
          annotation (Line(points={{-18,-50},{18,-50}},color={255,0,255}));
        connect(minSupTem.y, swi2.u1)
          annotation (Line(points={{-78,-10},{-2,-10},{-2,-42},{18,-42}},
            color={0,0,127}));
        connect(swi2.y, swi1.u3)
          annotation (Line(points={{42,-50},{50,-50},{50,-58},{78,-58}},
            color={0,0,127}));
        connect(TZonSetAve, TDea.u)
          annotation (Line(points={{-160,90},{-102,90}},
            color={0,0,127}));
        connect(uSupFan, swi3.u2)
          annotation (Line(points={{-160,-30},{-120,-30},{-120,10},{-60,10},{-60,0},
            {78,0}}, color={255,0,255}));
        connect(swi1.y, swi3.u1)
          annotation (Line(points={{102,-50},{110,-50},{110,-20},{68,-20},{68,8},{78,8}},
            color={0,0,127}));
        connect(TDea.y, swi3.u3)
          annotation (Line(points={{-78,90},{60,90},{60,-8},{78,-8}},
            color={0,0,127}));
        connect(intLesThr1.y, and1.u1)
          annotation (Line(points={{-18,-90},{18,-90}},
            color={255,0,255}));
        connect(intGreThr1.y, and1.u2)
          annotation (Line(points={{-18,-120},{0,-120},{0,-98},{18,-98}},
            color={255,0,255}));
        connect(intLesThr.y, and2.u1)
          annotation (Line(points={{-78,-50},{-42,-50}},color={255,0,255}));
        connect(intGreThr.y, and2.u2)
          annotation (Line(points={{-78,-80},{-60,-80},{-60,-58},{-42,-58}},
            color={255,0,255}));
        connect(uOpeMod, intLesThr.u)
          annotation (Line(points={{-160,-100},{-120,-100},{-120,-50},{-102,-50}},
            color={255,127,0}));
        connect(uOpeMod, intGreThr.u)
          annotation (Line(points={{-160,-100},{-120,-100},{-120,-80},{-102,-80}},
            color={255,127,0}));
        connect(uOpeMod, intLesThr1.u)
          annotation (Line(points={{-160,-100},{-60,-100},{-60,-90},{-42,-90}},
            color={255,127,0}));
        connect(uOpeMod, intGreThr1.u)
          annotation (Line(points={{-160,-100},{-120,-100},{-120,-120},{-42,-120}},
            color={255,127,0}));
        connect(uZonTemResReq, maxSupTemRes.numOfReq)
          annotation (Line(points={{-160,20},{-112,20},{-112,22},{-102,22}},
            color={255,127,0}));
        connect(uSupFan, maxSupTemRes.uDevSta)
          annotation (Line(points={{-160,-30},{-120,-30},{-120,38},{-102,38}},
            color={255,0,255}));
        connect(maxSupTemRes.y, lin.f1)
          annotation (Line(points={{-78,30},{-60,30},{-60,54},{18,54}},
            color={0,0,127}));
        connect(swi3.y, TSupSet)
          annotation (Line(points={{102,0},{160,0}},   color={0,0,127}));

        connect(lin.y, oveActTAirSup.u)
          annotation (Line(points={{42,50},{78,50}}, color={0,0,127}));
        connect(oveActTAirSup.y, swi2.u3) annotation (Line(points={{101,50},{
                120,50},{120,20},{14,20},{14,-58},{18,-58}}, color={0,0,127}));
      annotation (
        defaultComponentName = "conTSupSet",
        Icon(graphics={
              Rectangle(
              extent={{-100,-100},{100,100}},
              lineColor={0,0,127},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
              Text(
                extent={{-94,92},{-42,66}},
                lineColor={0,0,127},
                pattern=LinePattern.Dash,
                textString="TZonSetAve"),
              Text(
                extent={{-96,46},{-68,34}},
                lineColor={0,0,127},
                pattern=LinePattern.Dash,
                textString="TOut"),
              Text(
                extent={{-94,-22},{-14,-58}},
                lineColor={0,0,127},
                pattern=LinePattern.Dash,
                textString="uZonTemResReq"),
              Text(
                extent={{-94,12},{-48,-12}},
                lineColor={0,0,127},
                pattern=LinePattern.Dash,
                textString="uSupFan"),
              Text(
                extent={{-94,-70},{-50,-90}},
                lineColor={0,0,127},
                pattern=LinePattern.Dash,
                textString="uOpeMod"),
              Text(
                extent={{68,8},{96,-8}},
                lineColor={0,0,127},
                pattern=LinePattern.Dash,
                textString="TSupSet"),
              Text(
                extent={{-124,146},{96,108}},
                lineColor={0,0,255},
                textString="%name")}),
        Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-140,-140},{140,120}})),
        Documentation(info="<html>
<p>
Block that outputs the supply air temperature setpoint and the coil valve control
inputs for VAV system with multiple zones, implemented according to the ASHRAE
Guideline G36, PART 5.N.2 (Supply air temperature control).
</p>
<p>
The control loop is enabled when the supply air fan <code>uSupFan</code> is proven on,
and disabled and the output set to Deadband otherwise.
</p>
<p> The supply air temperature setpoint is computed as follows.</p>
<h4>Setpoints for <code>TSupSetMin</code>, <code>TSupSetMax</code>,
<code>TSupSetDes</code>, <code>TOutMin</code>, <code>TOutMax</code></h4>
<p>
The default range of outdoor air temperature (<code>TOutMin=16&deg;C</code>,
<code>TOutMax=21&deg;C</code>) used to reset the occupied mode <code>TSupSet</code>
was chosen to maximize economizer hours. It may be preferable to use a lower
range of outdoor air temperature (e.g. <code>TOutMin=13&deg;C</code>,
<code>TOutMax=18&deg;C</code>) to minimize fan energy.
</p>
<p>
The <code>TSupSetMin</code> variable is used during warm weather when little reheat
is expected to minimize fan energy. It should not be set too low or it may cause
excessive chilled water temperature reset requests which will reduce chiller
plant efficiency. It should be set no lower than the design coil leaving air
temperature.
</p>
<p>
The <code>TSupSetMax</code> variable is typically 18 &deg;C in mild and dry climate,
16 &deg;C or lower in humid climates. It should not typically be greater than
18 &deg;C since this may lead to excessive fan energy that can offset the mechanical
cooling savings from economizer operation.
</p>

<h4>During occupied mode (<code>uOpeMod=1</code>)</h4>
<p>
The <code>TSupSet</code> shall be reset from <code>TSupSetMin</code> when the outdoor
air temperature is <code>TOutMax</code> and above, proportionally up to
maximum supply temperature when the outdoor air temperature is <code>TOutMin</code> and
below. The maximum supply temperature shall be reset using trim and respond logic between
<code>TSupSetDes</code> and <code>TSupSetMax</code>. Parameters suggested for the
trim and respond logic are shown in the table below. They require adjustment
during the commissioning and tuning phase.
</p>

<table summary=\"summary\" border=\"1\">
<tr><th> Variable </th> <th> Value </th> <th> Definition </th> </tr>
<tr><td>Device</td><td>AHU Supply Fan</td> <td>Associated device</td></tr>
<tr><td>SP0</td><td><code>iniSet</code></td><td>Initial setpoint</td></tr>
<tr><td>SPmin</td><td><code>TSupSetDes</code></td><td>Minimum setpoint</td></tr>
<tr><td>SPmax</td><td><code>TSupSetMax</code></td><td>Maximum setpoint</td></tr>
<tr><td>Td</td><td><code>delTim</code></td><td>Delay timer</td></tr>
<tr><td>T</td><td><code>samplePeriod</code></td><td>Time step</td></tr>
<tr><td>I</td><td><code>numIgnReq</code></td><td>Number of ignored requests</td></tr>
<tr><td>R</td><td><code>uZonTemResReq</code></td><td>Number of requests</td></tr>
<tr><td>SPtrim</td><td><code>triAmo</code></td><td>Trim amount</td></tr>
<tr><td>SPres</td><td><code>resAmo</code></td><td>Respond amount</td></tr>
<tr><td>SPres_max</td><td><code>maxRes</code></td><td>Maximum response per time interval</td></tr>
</table>
<br/>

<p align=\"center\">
<img alt=\"Image of set point reset\"
src=\"modelica://Buildings/Resources/Images/Controls/OBC/ASHRAE/G36_PR1/AHUs/MultiZone/VAVSupTempSet.png\"/>
</p>

<h4>During Setup and Cool-down modes (<code>uOpeMod=2</code>, <code>uOpeMod=3</code>)</h4>
<p>
Supply air temperature setpoint <code>TSupSet</code> shall be <code>TSupSetMin</code>.
</p>
<h4>During Setback and Warmup modes (<code>uOpeMod=4</code>, <code>uOpeMod=5</code>)</h4>
<p>
Supply air temperature setpoint <code>TSupSet</code> shall be <code>TSupWarUpSetBac</code>.
</p>

<h4>Valves control</h4>
<p>
Supply air temperature shall be controlled to setpoint using a control loop whose
output is mapped to sequence the hot water valve or modulating electric heating
coil (if applicable) or chilled water valves.
</p>
</html>",
      revisions="<html>
<ul>
<li>
March 12, 2020, by Jianjun Hu:<br/>
Propagated supply temperature setpoint of warmup and setback mode.<br/>
This is for <a href=\"https://github.com/lbl-srg/modelica-buildings/issues/1829\">#1829</a>.
</li>
<li>
July 11, 2017, by Jianjun Hu:<br/>
First implementation.
</li>
</ul>
</html>"));
      end SupplyTemperatureOve;

      block SupplyFanOve "Block to control multi zone VAV AHU supply fan"

        parameter Boolean have_perZonRehBox = false
          "Check if there is any VAV-reheat boxes on perimeter zones"
          annotation(Dialog(group="System configuration"));
        parameter Boolean have_duaDucBox = false
          "Check if the AHU serves dual duct boxes"
          annotation(Dialog(group="System configuration"));
        parameter Boolean have_airFloMeaSta = false
          "Check if the AHU has AFMS (Airflow measurement station)"
          annotation(Dialog(group="System configuration"));
        parameter Real iniSet(
          final unit="Pa",
          final quantity="PressureDifference") = 120
          "Initial setpoint"
          annotation (Dialog(group="Trim and respond for pressure setpoint"));
        parameter Real minSet(
          final unit="Pa",
          final quantity="PressureDifference") = 25
          "Minimum setpoint"
          annotation (Dialog(group="Trim and respond for pressure setpoint"));
        parameter Real maxSet(
          final unit="Pa",
          final quantity="PressureDifference")
          "Maximum setpoint"
          annotation (Dialog(group="Trim and respond for pressure setpoint"));
        parameter Real delTim(
          final unit="s",
          final quantity="Time")= 600
         "Delay time after which trim and respond is activated"
          annotation (Dialog(group="Trim and respond for pressure setpoint"));
        parameter Real samplePeriod(
          final unit="s",
          final quantity="Time") = 120  "Sample period"
          annotation (Dialog(group="Trim and respond for pressure setpoint"));
        parameter Integer numIgnReq = 2
          "Number of ignored requests"
          annotation (Dialog(group="Trim and respond for pressure setpoint"));
        parameter Real triAmo(
          final unit="Pa",
          final quantity="PressureDifference") = -12.0
          "Trim amount"
          annotation (Dialog(group="Trim and respond for pressure setpoint"));
        parameter Real resAmo(
          final unit="Pa",
          final quantity="PressureDifference") = 15
          "Respond amount (must be opposite in to triAmo)"
          annotation (Dialog(group="Trim and respond for pressure setpoint"));
        parameter Real maxRes(
          final unit="Pa",
          final quantity="PressureDifference") = 32
          "Maximum response per time interval (same sign as resAmo)"
          annotation (Dialog(group="Trim and respond for pressure setpoint"));
        parameter Buildings.Controls.OBC.CDL.Types.SimpleController
          controllerType=Buildings.Controls.OBC.CDL.Types.SimpleController.PI "Type of controller"
          annotation (Dialog(group="Fan PID controller"));
        parameter Real k(final unit="1")=0.1
          "Gain of controller, normalized using maxSet"
          annotation (Dialog(group="Fan PID controller"));
        parameter Real Ti(
          final unit="s",
          final quantity="Time",
          min=0)=60
          "Time constant of integrator block"
          annotation (Dialog(group="Fan PID controller",
            enable=controllerType==Buildings.Controls.OBC.CDL.Types.SimpleController.PI
               or  controllerType==Buildings.Controls.OBC.CDL.Types.SimpleController.PID));
        parameter Real Td(
          final unit="s",
          final quantity="Time",
          final min=0) = 0.1
          "Time constant of derivative block"
          annotation (Dialog(group="Fan PID controller",
            enable=controllerType==Buildings.Controls.OBC.CDL.Types.SimpleController.PD
                or controllerType==Buildings.Controls.OBC.CDL.Types.SimpleController.PID));
        parameter Real yFanMax(min=0.1, max=1, unit="1") = 1
          "Maximum allowed fan speed"
          annotation (Dialog(group="Fan PID controller"));
        parameter Real yFanMin(min=0.1, max=1, unit="1") = 0.1
          "Lowest allowed fan speed if fan is on"
          annotation (Dialog(group="Fan PID controller"));

        Buildings.Controls.OBC.CDL.Interfaces.IntegerInput uOpeMod
         "System operation mode"
          annotation (Placement(transformation(extent={{-200,100},{-160,140}}),
              iconTransformation(extent={{-140,60},{-100,100}})));
        Buildings.Controls.OBC.CDL.Interfaces.RealInput ducStaPre(
          final unit="Pa",
          quantity="PressureDifference")
          "Measured duct static pressure"
          annotation (Placement(transformation(extent={{-200,-130},{-160,-90}}),
              iconTransformation(extent={{-140,-100},{-100,-60}})));
        Buildings.Controls.OBC.CDL.Interfaces.IntegerInput uZonPreResReq
          "Zone static pressure reset requests"
          annotation (Placement(transformation(extent={{-200,-80},{-160,-40}}),
            iconTransformation(extent={{-140,-50},{-100,-10}})));
        Buildings.Controls.OBC.CDL.Interfaces.BooleanOutput ySupFan "Supply fan on status"
          annotation (Placement(transformation(extent={{140,50},{180,90}}),
              iconTransformation(extent={{100,50},{140,90}})));
        Buildings.Controls.OBC.CDL.Interfaces.RealOutput ySupFanSpe(
          min=0,
          max=1,
          final unit="1") "Supply fan speed"
          annotation (Placement(transformation(extent={{140,-120},{180,-80}}),
              iconTransformation(extent={{100,-20},{140,20}})));

        Buildings.Controls.OBC.ASHRAE.G36_PR1.Generic.SetPoints.TrimAndRespond staPreSetRes(
          final iniSet=iniSet,
          final minSet=minSet,
          final maxSet=maxSet,
          final delTim=delTim,
          final samplePeriod=samplePeriod,
          final numIgnReq=numIgnReq,
          final triAmo=triAmo,
          final resAmo=resAmo,
          final maxRes=maxRes) "Static pressure setpoint reset using trim and respond logic"
          annotation (Placement(transformation(extent={{-130,-48},{-110,-28}})));
        Buildings.Controls.OBC.CDL.Continuous.LimPID conSpe(
          final controllerType=controllerType,
          final k=k,
          final Ti=Ti,
          final Td=Td,
          final yMax=yFanMax,
          final yMin=yFanMin,
          reset=Buildings.Controls.OBC.CDL.Types.Reset.Parameter,
          y_reset=yFanMin) "Supply fan speed control"
          annotation (Placement(transformation(extent={{-40,-80},{-20,-60}})));

        Buildings.Utilities.IO.SignalExchange.Overwrite oveActdpAir(description=
             "Supply air dp setpoint", u(
            unit="Pa",
            min=25,
            max=410)) "Overwrite the supply air dp setpoint"
          annotation (Placement(transformation(extent={{-66,-48},{-46,-28}})));
      protected
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant zerSpe(k=0)
          "Zero fan speed when it becomes OFF"
          annotation (Placement(transformation(extent={{20,-90},{40,-70}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi
          "If fan is OFF, fan speed outputs to zero"
          annotation (Placement(transformation(extent={{80,-90},{100,-110}})));
        Buildings.Controls.OBC.CDL.Logical.Or or1
          "Check whether supply fan should be ON"
          annotation (Placement(transformation(extent={{80,60},{100,80}})));
        Buildings.Controls.OBC.CDL.Logical.Or or2 if have_perZonRehBox
          "Setback or warmup mode"
          annotation (Placement(transformation(extent={{20,30},{40,50}})));
        Buildings.Controls.OBC.CDL.Logical.Or3 or3
          "Cool-down or setup or occupied mode"
          annotation (Placement(transformation(extent={{20,90},{40,110}})));
        Buildings.Controls.OBC.CDL.Logical.Sources.Constant con(
          k=false) if not have_perZonRehBox
          "Constant true"
          annotation (Placement(transformation(extent={{20,0},{40,20}})));
        Buildings.Controls.OBC.CDL.Integers.Sources.Constant conInt(
          k=Buildings.Controls.OBC.ASHRAE.G36_PR1.Types.OperationModes.coolDown)
          "Cool down mode"
          annotation (Placement(transformation(extent={{-120,120},{-100,140}})));
        Buildings.Controls.OBC.CDL.Integers.Sources.Constant conInt4(
          k=Buildings.Controls.OBC.ASHRAE.G36_PR1.Types.OperationModes.warmUp)
          "Warm-up mode"
          annotation (Placement(transformation(extent={{-120,0},{-100,20}})));
        Buildings.Controls.OBC.CDL.Integers.Sources.Constant conInt1(
          k=Buildings.Controls.OBC.ASHRAE.G36_PR1.Types.OperationModes.setUp)
          "Set up mode"
          annotation (Placement(transformation(extent={{-120,90},{-100,110}})));
        Buildings.Controls.OBC.CDL.Integers.Sources.Constant conInt2(
          k=Buildings.Controls.OBC.ASHRAE.G36_PR1.Types.OperationModes.occupied)
          "Occupied mode"
          annotation (Placement(transformation(extent={{-120,60},{-100,80}})));
        Buildings.Controls.OBC.CDL.Integers.Sources.Constant conInt3(
          k=Buildings.Controls.OBC.ASHRAE.G36_PR1.Types.OperationModes.setBack)
          "Set back mode"
          annotation (Placement(transformation(extent={{-120,30},{-100,50}})));
        Buildings.Controls.OBC.CDL.Integers.Equal intEqu
          "Check if current operation mode is cool-down mode"
          annotation (Placement(transformation(extent={{-60,120},{-40,140}})));
        Buildings.Controls.OBC.CDL.Integers.Equal intEqu1
          "Check if current operation mode is setup mode"
          annotation (Placement(transformation(extent={{-60,90},{-40,110}})));
        Buildings.Controls.OBC.CDL.Integers.Equal intEqu2
          "Check if current operation mode is occupied mode"
          annotation (Placement(transformation(extent={{-60,60},{-40,80}})));
        Buildings.Controls.OBC.CDL.Integers.Equal intEqu3
          "Check if current operation mode is setback mode"
          annotation (Placement(transformation(extent={{-60,30},{-40,50}})));
        Buildings.Controls.OBC.CDL.Integers.Equal intEqu4
          "Check if current operation mode is warmup mode"
          annotation (Placement(transformation(extent={{-60,0},{-40,20}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant gaiNor(
          final k=maxSet)
          "Gain for normalization of controller input"
          annotation (Placement(transformation(extent={{-130,-100},{-110,-80}})));
        Buildings.Controls.OBC.CDL.Continuous.Division norPSet
          "Normalization for pressure set point"
          annotation (Placement(transformation(extent={{-70,-80},{-50,-60}})));
        Buildings.Controls.OBC.CDL.Continuous.Division norPMea
          "Normalization of pressure measurement"
          annotation (Placement(transformation(extent={{-70,-120},{-50,-100}})));
        Buildings.Controls.OBC.CDL.Discrete.FirstOrderHold firOrdHol(
          final samplePeriod=samplePeriod)
          "Extrapolation through the values of the last two sampled input signals"
          annotation (Placement(transformation(extent={{-100,-48},{-80,-28}})));

      equation
        connect(or2.y, or1.u2)
          annotation (Line(points={{42,40},{60,40},{60,62},{78,62}},
            color={255,0,255}));
        connect(or1.y, ySupFan)
          annotation (Line(points={{102,70},{160,70}},
            color={255,0,255}));
        connect(or1.y, staPreSetRes.uDevSta)
          annotation (Line(points={{102,70},{120,70},{120,-8},{-150,-8},{-150,
                -30},{-132,-30}},
                           color={255,0,255}));
        connect(or1.y, swi.u2)
          annotation (Line(points={{102,70},{120,70},{120,-8},{0,-8},{0,-100},{78,-100}},
            color={255,0,255}));
        connect(conSpe.y, swi.u1)
          annotation (Line(points={{-18,-70},{-4,-70},{-4,-108},{78,-108}},
            color={0,0,127}));
        connect(zerSpe.y, swi.u3)
          annotation (Line(points={{42,-80},{60,-80},{60,-92},{78,-92}},
            color={0,0,127}));
        connect(swi.y, ySupFanSpe)
          annotation (Line(points={{102,-100},{160,-100}},
            color={0,0,127}));
        connect(uZonPreResReq, staPreSetRes.numOfReq)
          annotation (Line(points={{-180,-60},{-148,-60},{-148,-46},{-132,-46}},
            color={255,127,0}));
        connect(con.y, or1.u2)
          annotation (Line(points={{42,10},{60,10},{60,62},{78,62}},
            color={255,0,255}));
        connect(intEqu.y, or3.u1)
          annotation (Line(points={{-38,130},{0,130},{0,108},{18,108}},
            color={255,0,255}));
        connect(intEqu2.y, or3.u3)
          annotation (Line(points={{-38,70},{0,70},{0,92},{18,92}},
            color={255,0,255}));
        connect(intEqu1.y, or3.u2)
          annotation (Line(points={{-38,100},{18,100}}, color={255,0,255}));
        connect(conInt.y, intEqu.u2)
          annotation (Line(points={{-98,130},{-90,130},{-90,122},{-62,122}},
            color={255,127,0}));
        connect(conInt1.y, intEqu1.u2)
          annotation (Line(points={{-98,100},{-90,100},{-90,92},{-62,92}},
            color={255,127,0}));
        connect(conInt2.y, intEqu2.u2)
          annotation (Line(points={{-98,70},{-90,70},{-90,62},{-62,62}},
            color={255,127,0}));
        connect(conInt3.y, intEqu3.u2)
          annotation (Line(points={{-98,40},{-90,40},{-90,32},{-62,32}},
            color={255,127,0}));
        connect(conInt4.y, intEqu4.u2)
          annotation (Line(points={{-98,10},{-90,10},{-90,2},{-62,2}},
            color={255,127,0}));
        connect(uOpeMod, intEqu.u1)
          annotation (Line(points={{-180,120},{-140,120},{-140,150},{-80,150},{-80,130},
            {-62,130}}, color={255,127,0}));
        connect(uOpeMod, intEqu1.u1)
          annotation (Line(points={{-180,120},{-140,120},{-140,150},{-80,150},{-80,100},
            {-62,100}}, color={255,127,0}));
        connect(uOpeMod, intEqu2.u1)
          annotation (Line(points={{-180,120},{-140,120},{-140,150},{-80,150},
            {-80,70},{-62,70}}, color={255,127,0}));
        connect(uOpeMod, intEqu3.u1)
          annotation (Line(points={{-180,120},{-140,120},{-140,150},{-80,150},
            {-80,40},{-62,40}}, color={255,127,0}));
        connect(uOpeMod, intEqu4.u1)
          annotation (Line(points={{-180,120},{-140,120},{-140,150},{-80,150},
            {-80,10},{-62,10}}, color={255,127,0}));
        connect(or3.y, or1.u1)
          annotation (Line(points={{42,100},{60,100},{60,70},{78,70}},
            color={255,0,255}));
        connect(intEqu3.y, or2.u1)
          annotation (Line(points={{-38,40},{18,40}}, color={255,0,255}));
        connect(intEqu4.y, or2.u2)
          annotation (Line(points={{-38,10},{0,10},{0,32},{18,32}},
            color={255,0,255}));
        connect(norPSet.y, conSpe.u_s)
          annotation (Line(points={{-48,-70},{-42,-70}}, color={0,0,127}));
        connect(norPMea.y, conSpe.u_m)
          annotation (Line(points={{-48,-110},{-30,-110},{-30,-82}}, color={0,0,127}));
        connect(staPreSetRes.y, firOrdHol.u)
          annotation (Line(points={{-108,-38},{-102,-38}}, color={0,0,127}));
        connect(conSpe.trigger, or1.y)
          annotation (Line(points={{-36,-82},{-36,-100},{0,-100},{0,-8},{120,-8},
                {120,70},{102,70}},
                                color={255,0,255}));
        connect(gaiNor.y, norPSet.u2) annotation (Line(points={{-108,-90},{-92,-90},{-92,
                -76},{-72,-76}}, color={0,0,127}));
        connect(ducStaPre, norPMea.u1) annotation (Line(points={{-180,-110},{-80,-110},
                {-80,-104},{-72,-104}}, color={0,0,127}));
        connect(gaiNor.y, norPMea.u2) annotation (Line(points={{-108,-90},{-92,-90},{-92,
                -116},{-72,-116}}, color={0,0,127}));

        connect(firOrdHol.y, oveActdpAir.u)
          annotation (Line(points={{-78,-38},{-68,-38}}, color={0,0,127}));
        connect(oveActdpAir.y, norPSet.u1) annotation (Line(points={{-45,-38},{
                -40,-38},{-40,-52},{-92,-52},{-92,-64},{-72,-64}}, color={0,0,
                127}));
      annotation (
        defaultComponentName="conSupFan",
        Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-160,-140},{140,160}}),
              graphics={
              Rectangle(
                extent={{-156,-22},{134,-128}},
                lineColor={0,0,0},
                fillColor={215,215,215},
                fillPattern=FillPattern.Solid,
                pattern=LinePattern.None),
              Rectangle(
                extent={{-156,156},{134,-6}},
                lineColor={0,0,0},
                fillColor={215,215,215},
                fillPattern=FillPattern.Solid,
                pattern=LinePattern.None),
              Text(
                extent={{42,156},{124,134}},
                lineColor={0,0,255},
                fillColor={215,215,215},
                fillPattern=FillPattern.Solid,
                horizontalAlignment=TextAlignment.Left,
                textString="Check current operation mode"),
              Text(
                extent={{54,-26},{124,-38}},
                lineColor={0,0,255},
                fillColor={215,215,215},
                fillPattern=FillPattern.Solid,
                horizontalAlignment=TextAlignment.Left,
                textString="Reset pressure setpoint"),
              Text(
                extent={{-34,-106},{20,-136}},
                lineColor={0,0,255},
                fillColor={215,215,215},
                fillPattern=FillPattern.Solid,
                textString="Control fan speed"),
              Text(
                extent={{42,142},{96,126}},
                lineColor={0,0,255},
                fillColor={215,215,215},
                fillPattern=FillPattern.Solid,
                horizontalAlignment=TextAlignment.Left,
                textString="Check fan on or off")}),
        Icon(graphics={
              Text(
                extent={{-102,140},{96,118}},
                lineColor={0,0,255},
                textString="%name"),
                     Rectangle(
                extent={{-100,100},{100,-100}},
                lineColor={0,0,0},
                fillColor={223,211,169},
                fillPattern=FillPattern.Solid),
              Text(
                extent={{-96,90},{-54,70}},
                lineColor={0,0,127},
                textString="uOpeMod"),
              Text(
                extent={{-96,-16},{-44,-44}},
                lineColor={0,0,127},
                textString="uZonPreResReq"),
              Text(
                extent={{-96,-70},{-54,-90}},
                lineColor={0,0,127},
                textString="ducStaPre"),
              Text(
                extent={{54,-60},{96,-80}},
                lineColor={0,0,127},
                textString="sumVDis_flow"),
              Text(
                extent={{52,10},{94,-10}},
                lineColor={0,0,127},
                textString="yFanSpe"),
              Text(
                extent={{52,78},{94,58}},
                lineColor={0,0,127},
                textString="ySupFan")}),
        Documentation(info="<html>
<p>
Supply fan control for a multi zone VAV AHU according to
ASHRAE guideline G36, PART 5.N.1 (Supply fan control).
</p>
<h4>Supply fan start/stop</h4>
<ul>
<li>Supply fan shall run when system is in the Cool-down, Setup, or Occupied mode</li>
<li>If there are any VAV-reheat boxes on perimeter zones, supply fan shall also
run when system is in Setback or Warmup mode;</li>
<li>If the AHU does not serve dual duct boxes
that do not have hot-duct inlet airflow sensors (<code>have_duaDucBox=true</code>)
or the AHU does not have airflow measurement station (<code>have_airFloMeaSta=false</code>),
sum the current airflow rate from the VAV boxes and output to a software point.</li>
</ul>
<h4>Static pressure setpoint reset</h4>
<p>
Static pressure setpoint shall be reset using trim-respond logic using following
parameters as a starting point:
</p>
<table summary=\"summary\" border=\"1\">
<tr><th> Variable </th> <th> Value </th> <th> Definition </th> </tr>
<tr><td>Device</td><td>AHU Supply Fan</td> <td>Associated device</td></tr>
<tr><td>SP0</td><td><code>iniSet</code></td><td>Initial setpoint</td></tr>
<tr><td>SPmin</td><td><code>minSet</code></td><td>Minimum setpoint</td></tr>
<tr><td>SPmax</td><td><code>maxSet</code></td><td>Maximum setpoint</td></tr>
<tr><td>Td</td><td><code>delTim</code></td><td>Delay timer</td></tr>
<tr><td>T</td><td><code>samplePeriod</code></td><td>Time step</td></tr>
<tr><td>I</td><td><code>numIgnReq</code></td><td>Number of ignored requests</td></tr>
<tr><td>R</td><td><code>uZonPreResReq</code></td><td>Number of requests</td></tr>
<tr><td>SPtrim</td><td><code>triAmo</code></td><td>Trim amount</td></tr>
<tr><td>SPres</td><td><code>resAmo</code></td><td>Respond amount</td></tr>
<tr><td>SPres_max</td><td><code>maxRes</code></td><td>Maximum response per time interval</td></tr>
</table>
<br/>
<h4>Static pressure control</h4>
<p>
Supply fan speed is controlled with a PI controller to maintain duct static pressure at setpoint
when the fan is proven on. The setpoint for the PI controller and the measured
duct static pressure are normalized with the maximum design static presssure
<code>maxSet</code>.
Where the zone groups served by the system are small,
provide multiple sets of gains that are used in the control loop as a function
of a load indicator (such as supply fan airflow rate, the area of the zone groups
that are occupied, etc.).
</p>
</html>",       revisions="<html>
<ul>
<li>
March 12, 2020, by Jianjun Hu:<br/>
Removed the sum of flow rate as it is not used in any other sequences.<br/>
This is for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/1829\">issue 1829</a>.
</li>
<li>
January 7, 2020, by Michael Wetter:<br/>
Reformulated to avoid relying on the <code>final</code> keyword.<br/>
This is for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/1701\">issue 1701</a>.
</li>
<li>
October 14, 2017, by Michael Wetter:<br/>
Added normalization of pressure set point and measurement as the measured
quantity is a few hundred Pascal.
</li>
<li>
August 15, 2017, by Jianjun Hu:<br/>
First implementation.
</li>
</ul>
</html>"));
      end SupplyFanOve;

      package Examples "Example models to test the components"
          extends Modelica.Icons.ExamplesPackage;
        model OperationModes "Test model for operation modes"
            extends Modelica.Icons.Example;
          import ModelicaVAV = FiveZoneAir.VAVReheat;
          ModelicaVAV.Controls.ModeSelector operationModes
            annotation (Placement(transformation(extent={{-40,-20},{-20,0}})));
          Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow preHeaFlo
            annotation (Placement(transformation(extent={{90,-60},{110,-40}})));
          Modelica.Thermal.HeatTransfer.Sources.FixedTemperature fixTem(T=273.15)
            annotation (Placement(transformation(extent={{-40,90},{-20,110}})));
          Modelica.Thermal.HeatTransfer.Components.HeatCapacitor cap(C=20000, T(fixed=
                  true))
            annotation (Placement(transformation(extent={{40,100},{60,120}})));
          Modelica.Thermal.HeatTransfer.Components.ThermalConductor con(G=1)
            annotation (Placement(transformation(extent={{0,90},{20,110}})));
          Modelica.Blocks.Logical.Switch switch1
            annotation (Placement(transformation(extent={{60,-60},{80,-40}})));
          Modelica.Blocks.Sources.Constant on(k=200)
            annotation (Placement(transformation(extent={{20,-40},{40,-20}})));
          Modelica.Blocks.Sources.Constant off(k=0)
            annotation (Placement(transformation(extent={{-60,-80},{-40,-60}})));
          Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor
            annotation (Placement(transformation(extent={{100,110},{120,130}})));
          Modelica.Blocks.Sources.RealExpression TRooSetHea(
            y=if mode.y == Integer(ModelicaVAV.Controls.OperationModes.occupied)
              then 293.15 else 287.15)
            annotation (Placement(transformation(extent={{-160,40},{-140,60}})));
          Modelica.Blocks.Sources.Constant TCoiHea(k=283.15)
            "Temperature after heating coil"
            annotation (Placement(transformation(extent={{-160,-40},{-140,-20}})));
          ModelicaVAV.Controls.ControlBus controlBus
            annotation (Placement(transformation(extent={{-60,30},{-40,50}})));
          Modelica.Blocks.Routing.IntegerPassThrough mode "Outputs the control mode"
            annotation (Placement(transformation(extent={{0,20},{20,40}})));
          Modelica.Blocks.Sources.BooleanExpression modSel(
            y=mode.y == Integer(ModelicaVAV.Controls.OperationModes.unoccupiedNightSetBack) or
              mode.y == Integer(ModelicaVAV.Controls.OperationModes.unoccupiedWarmUp))
            annotation (Placement(transformation(extent={{-20,-60},{0,-40}})));
          Modelica.Blocks.Sources.Constant TOut(k=283.15) "Outside temperature"
            annotation (Placement(transformation(extent={{-160,-80},{-140,-60}})));
          Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor1
            annotation (Placement(transformation(extent={{100,142},{120,162}})));
          Modelica.Blocks.Sources.BooleanExpression modSel1(
            y=mode.y == Integer(ModelicaVAV.Controls.OperationModes.occupied))
            annotation (Placement(transformation(extent={{-20,-130},{0,-110}})));
          Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow preHeaFlo1
            annotation (Placement(transformation(extent={{112,-130},{132,-110}})));
          Buildings.Controls.Continuous.LimPID PID(initType=Modelica.Blocks.Types.InitPID.InitialState)
            annotation (Placement(transformation(extent={{-80,-130},{-60,-110}})));
          Modelica.Blocks.Logical.Switch switch2
            annotation (Placement(transformation(extent={{20,-130},{40,-110}})));
          Modelica.Blocks.Math.Gain gain(k=200)
            annotation (Placement(transformation(extent={{62,-130},{82,-110}})));
          Buildings.Controls.SetPoints.OccupancySchedule occSch "Occupancy schedule"
            annotation (Placement(transformation(extent={{-160,80},{-140,100}})));
        equation
          connect(fixTem.port, con.port_a) annotation (Line(
              points={{-20,100},{0,100}},
              color={191,0,0},
              smooth=Smooth.None));
          connect(preHeaFlo.port, cap.port) annotation (Line(
              points={{110,-50},{120,-50},{120,100},{50,100}},
              color={191,0,0},
              smooth=Smooth.None));
          connect(con.port_b, cap.port) annotation (Line(
              points={{20,100},{50,100}},
              color={191,0,0},
              smooth=Smooth.None));
          connect(switch1.y, preHeaFlo.Q_flow) annotation (Line(
              points={{81,-50},{90,-50}},
              color={0,0,127},
              smooth=Smooth.None));
          connect(on.y, switch1.u1) annotation (Line(
              points={{41,-30},{48,-30},{48,-42},{58,-42}},
              color={0,0,127},
              smooth=Smooth.None));
          connect(off.y, switch1.u3) annotation (Line(
              points={{-39,-70},{48,-70},{48,-58},{58,-58}},
              color={0,0,127},
              smooth=Smooth.None));
          connect(cap.port, temperatureSensor.port) annotation (Line(
              points={{50,100},{64,100},{64,120},{100,120}},
              color={191,0,0},
              smooth=Smooth.None));
          connect(controlBus, operationModes.cb) annotation (Line(
              points={{-50,40},{-50,-3.18182},{-36.8182,-3.18182}},
              color={255,204,51},
              thickness=0.5,
              smooth=Smooth.None));
          connect(temperatureSensor.T, controlBus.TRooMin) annotation (Line(
              points={{120,120},{166,120},{166,60},{-50,60},{-50,40}},
              color={0,0,127},
              smooth=Smooth.None), Text(
              textString="%second",
              index=1,
              extent={{6,3},{6,3}}));
          connect(TCoiHea.y, controlBus.TCoiHeaOut) annotation (Line(
              points={{-139,-30},{-78,-30},{-78,40},{-50,40}},
              color={0,0,127},
              smooth=Smooth.None), Text(
              textString="%second",
              index=1,
              extent={{6,3},{6,3}}));
          connect(controlBus.controlMode, mode.u) annotation (Line(
              points={{-50,40},{-30,40},{-30,30},{-2,30}},
              color={255,204,51},
              thickness=0.5,
              smooth=Smooth.None), Text(
              textString="%first",
              index=-1,
              extent={{-6,3},{-6,3}}));
          connect(modSel.y, switch1.u2) annotation (Line(
              points={{1,-50},{58,-50}},
              color={255,0,255},
              smooth=Smooth.None));
          connect(TOut.y, controlBus.TOut) annotation (Line(
              points={{-139,-70},{-72,-70},{-72,40},{-50,40}},
              color={0,0,127},
              smooth=Smooth.None), Text(
              textString="%second",
              index=1,
              extent={{6,3},{6,3}}));
          connect(cap.port, temperatureSensor1.port) annotation (Line(
              points={{50,100},{64,100},{64,152},{100,152}},
              color={191,0,0},
              smooth=Smooth.None));
          connect(temperatureSensor1.T, controlBus.TRooAve) annotation (Line(
              points={{120,152},{166,152},{166,60},{-50,60},{-50,40}},
              color={0,0,127},
              smooth=Smooth.None), Text(
              textString="%second",
              index=1,
              extent={{6,3},{6,3}}));
          connect(TRooSetHea.y, PID.u_s)
                                      annotation (Line(
              points={{-139,50},{-110,50},{-110,-120},{-82,-120}},
              color={0,0,127},
              smooth=Smooth.None));
          connect(temperatureSensor.T, PID.u_m) annotation (Line(
              points={{120,120},{166,120},{166,-150},{-70,-150},{-70,-132}},
              color={0,0,127},
              smooth=Smooth.None));
          connect(modSel1.y, switch2.u2) annotation (Line(
              points={{1,-120},{18,-120}},
              color={255,0,255},
              smooth=Smooth.None));
          connect(off.y, switch2.u3) annotation (Line(
              points={{-39,-70},{-30,-70},{-30,-132},{10,-132},{10,-128},{18,-128}},
              color={0,0,127},
              smooth=Smooth.None));
          connect(preHeaFlo1.port, cap.port) annotation (Line(
              points={{132,-120},{148,-120},{148,100},{50,100}},
              color={191,0,0},
              smooth=Smooth.None));
          connect(PID.y, switch2.u1) annotation (Line(
              points={{-59,-120},{-38,-120},{-38,-100},{8,-100},{8,-112},{18,-112}},
              color={0,0,127},
              smooth=Smooth.None));
          connect(gain.y, preHeaFlo1.Q_flow) annotation (Line(
              points={{83,-120},{112,-120}},
              color={0,0,127},
              smooth=Smooth.None));
          connect(switch2.y, gain.u) annotation (Line(
              points={{41,-120},{60,-120}},
              color={0,0,127},
              smooth=Smooth.None));
          connect(occSch.tNexOcc, controlBus.dTNexOcc) annotation (Line(
              points={{-139,96},{-50,96},{-50,40}},
              color={0,0,127},
              smooth=Smooth.None), Text(
              textString="%second",
              index=1,
              extent={{6,3},{6,3}}));
          connect(TRooSetHea.y, controlBus.TRooSetHea) annotation (Line(
              points={{-139,50},{-100,50},{-100,40},{-50,40}},
              color={0,0,127},
              smooth=Smooth.None), Text(
              textString="%second",
              index=1,
              extent={{6,3},{6,3}}));
          connect(occSch.occupied, controlBus.occupied) annotation (Line(
              points={{-139,84},{-50,84},{-50,40}},
              color={255,0,255},
              smooth=Smooth.None), Text(
              textString="%second",
              index=1,
              extent={{6,3},{6,3}}));
          annotation (Diagram(coordinateSystem(preserveAspectRatio=true, extent={{-200,
                    -200},{200,200}})),
                __Dymola_Commands(file="modelica://Buildings/Resources/Scripts/Dymola/Examples/VAVReheat/Controls/Examples/OperationModes.mos"
                "Simulate and plot"),
            experiment(
              StopTime=172800,
              Tolerance=1e-6),
            Documentation(info="<html>
<p>
This model tests the transition between the different modes of operation of the HVAC system.
</p>
</html>"));
        end OperationModes;

        model RoomVAV "Test model for the room VAV controller"
          extends Modelica.Icons.Example;

          FiveZoneAir.VAVReheat.Controls.RoomVAV vavBoxCon
            "VAV terminal unit single maximum controller"
            annotation (Placement(transformation(extent={{40,-10},{60,10}})));
          Buildings.Controls.OBC.CDL.Continuous.Sources.Constant heaSet(k=273.15 + 21)
            "Heating setpoint"
            annotation (Placement(transformation(extent={{-40,60},{-20,80}})));
          Buildings.Controls.OBC.CDL.Continuous.Sources.Constant cooSet(k=273.15 + 22)
            "Cooling setpoint"
            annotation (Placement(transformation(extent={{-40,10},{-20,30}})));
          Buildings.Controls.OBC.CDL.Continuous.Sources.Ramp ram(
            height=4,
            duration=3600,
            offset=-4) "Ramp source"
            annotation (Placement(transformation(extent={{-80,-40},{-60,-20}})));
          Buildings.Controls.OBC.CDL.Continuous.Sources.Sine sin(
            amplitude=1,
            freqHz=1/3600,
            offset=273.15 + 23.5) "Sine source"
            annotation (Placement(transformation(extent={{-80,-80},{-60,-60}})));
          Buildings.Controls.OBC.CDL.Continuous.Add rooTem "Room temperature"
            annotation (Placement(transformation(extent={{-20,-60},{0,-40}})));

        equation
          connect(rooTem.y, vavBoxCon.TRoo) annotation (Line(points={{2,-50},{20,-50},{20,
                  -7},{39,-7}}, color={0,0,127}));
          connect(cooSet.y, vavBoxCon.TRooCooSet)
            annotation (Line(points={{-18,20},{0,20},{0,0},{38,0}}, color={0,0,127}));
          connect(heaSet.y, vavBoxCon.TRooHeaSet) annotation (Line(points={{-18,70},{20,
                  70},{20,7},{38,7}}, color={0,0,127}));
          connect(sin.y, rooTem.u2) annotation (Line(points={{-58,-70},{-40,-70},{-40,-56},
                  {-22,-56}}, color={0,0,127}));
          connect(ram.y, rooTem.u1) annotation (Line(points={{-58,-30},{-40,-30},{-40,-44},
                  {-22,-44}}, color={0,0,127}));

        annotation (
          __Dymola_Commands(file="modelica://Buildings/Resources/Scripts/Dymola/Examples/VAVReheat/Controls/Examples/RoomVAV.mos"
                "Simulate and plot"),
            experiment(StopTime=3600, Tolerance=1e-6),
            Documentation(info="<html>
<p>
This model tests the VAV box contoller of transition from heating control to cooling
control.
</p>
</html>"),
        Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
                coordinateSystem(preserveAspectRatio=false)));
        end RoomVAV;
      end Examples;

    end Controls;

    package ThermalZones "Package with models for the thermal zones"
    extends Modelica.Icons.VariantsPackage;
      model Floor "Model of a floor of the building"
        replaceable package Medium = Modelica.Media.Interfaces.PartialMedium
          "Medium model for air" annotation (choicesAllMatching=true);

        parameter Boolean use_windPressure=true
          "Set to true to enable wind pressure";

        parameter Buildings.HeatTransfer.Types.InteriorConvection intConMod=Buildings.HeatTransfer.Types.InteriorConvection.Temperature
          "Convective heat transfer model for room-facing surfaces of opaque constructions";
        parameter Modelica.SIunits.Angle lat "Latitude";
        parameter Real winWalRat(
          min=0.01,
          max=0.99) = 0.33 "Window to wall ratio for exterior walls";
        parameter Modelica.SIunits.Length hWin = 1.5 "Height of windows";
        parameter Buildings.HeatTransfer.Data.Solids.Plywood matFur(x=0.15, nStaRef=5)
          "Material for furniture"
          annotation (Placement(transformation(extent={{140,460},{160,480}})));
        parameter Buildings.HeatTransfer.Data.Resistances.Carpet matCar "Carpet"
          annotation (Placement(transformation(extent={{180,460},{200,480}})));
        parameter Buildings.HeatTransfer.Data.Solids.Concrete matCon(
          x=0.1,
          k=1.311,
          c=836,
          nStaRef=5) "Concrete"
          annotation (Placement(transformation(extent={{140,430},{160,450}})));
        parameter Buildings.HeatTransfer.Data.Solids.Plywood matWoo(
          x=0.01,
          k=0.11,
          d=544,
          nStaRef=1) "Wood for exterior construction"
          annotation (Placement(transformation(extent={{140,400},{160,420}})));
        parameter Buildings.HeatTransfer.Data.Solids.Generic matIns(
          x=0.087,
          k=0.049,
          c=836.8,
          d=265,
          nStaRef=5) "Steelframe construction with insulation"
          annotation (Placement(transformation(extent={{180,400},{200,420}})));
        parameter Buildings.HeatTransfer.Data.Solids.GypsumBoard matGyp(
          x=0.0127,
          k=0.16,
          c=830,
          d=784,
          nStaRef=2) "Gypsum board"
          annotation (Placement(transformation(extent={{138,372},{158,392}})));
        parameter Buildings.HeatTransfer.Data.Solids.GypsumBoard matGyp2(
          x=0.025,
          k=0.16,
          c=830,
          d=784,
          nStaRef=2) "Gypsum board"
          annotation (Placement(transformation(extent={{178,372},{198,392}})));
        parameter Buildings.HeatTransfer.Data.OpaqueConstructions.Generic conExtWal(final
            nLay=3, material={matWoo,matIns,matGyp}) "Exterior construction"
          annotation (Placement(transformation(extent={{280,460},{300,480}})));
        parameter Buildings.HeatTransfer.Data.OpaqueConstructions.Generic conIntWal(final
            nLay=1, material={matGyp2}) "Interior wall construction"
          annotation (Placement(transformation(extent={{320,460},{340,480}})));
        parameter Buildings.HeatTransfer.Data.OpaqueConstructions.Generic conFlo(final
            nLay=1, material={matCon}) "Floor construction (opa_a is carpet)"
          annotation (Placement(transformation(extent={{280,420},{300,440}})));
        parameter Buildings.HeatTransfer.Data.OpaqueConstructions.Generic conFur(final
            nLay=1, material={matFur}) "Construction for internal mass of furniture"
          annotation (Placement(transformation(extent={{320,420},{340,440}})));
        parameter Buildings.HeatTransfer.Data.Solids.Plywood matCarTra(
          k=0.11,
          d=544,
          nStaRef=1,
          x=0.215/0.11) "Wood for floor"
          annotation (Placement(transformation(extent={{102,460},{122,480}})));
        parameter Buildings.HeatTransfer.Data.GlazingSystems.DoubleClearAir13Clear glaSys(
          UFra=2,
          shade=Buildings.HeatTransfer.Data.Shades.Gray(),
          haveInteriorShade=false,
          haveExteriorShade=false) "Data record for the glazing system"
          annotation (Placement(transformation(extent={{240,460},{260,480}})));
        parameter Real kIntNor(min=0, max=1) = 1
          "Gain factor to scale internal heat gain in north zone";
        constant Modelica.SIunits.Height hRoo=2.74 "Room height";

        parameter Boolean sampleModel = false
          "Set to true to time-sample the model, which can give shorter simulation time if there is already time sampling in the system model"
          annotation (
            Evaluate=true,
            Dialog(tab="Experimental (may be changed in future releases)"));

        Buildings.ThermalZones.Detailed.MixedAir sou(
          redeclare package Medium = Medium,
          lat=lat,
          AFlo=568.77/hRoo,
          hRoo=hRoo,
          nConExt=0,
          nConExtWin=1,
          datConExtWin(
            layers={conExtWal},
            A={49.91*hRoo},
            glaSys={glaSys},
            wWin={winWalRat/hWin*49.91*hRoo},
            each hWin=hWin,
            fFra={0.1},
            til={Buildings.Types.Tilt.Wall},
            azi={Buildings.Types.Azimuth.S}),
          nConPar=2,
          datConPar(
            layers={conFlo,conFur},
            A={568.77/hRoo,414.68},
            til={Buildings.Types.Tilt.Floor,Buildings.Types.Tilt.Wall}),
          nConBou=3,
          datConBou(
            layers={conIntWal,conIntWal,conIntWal},
            A={6.47,40.76,6.47}*hRoo,
            til={Buildings.Types.Tilt.Wall, Buildings.Types.Tilt.Wall, Buildings.Types.Tilt.Wall}),
          nSurBou=0,
          nPorts=5,
          intConMod=intConMod,
          energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
          final sampleModel=sampleModel) "South zone"
          annotation (Placement(transformation(extent={{144,-44},{184,-4}})));
        Buildings.ThermalZones.Detailed.MixedAir eas(
          redeclare package Medium = Medium,
          lat=lat,
          AFlo=360.0785/hRoo,
          hRoo=hRoo,
          nConExt=0,
          nConExtWin=1,
          datConExtWin(
            layers={conExtWal},
            A={33.27*hRoo},
            glaSys={glaSys},
            wWin={winWalRat/hWin*33.27*hRoo},
            each hWin=hWin,
            fFra={0.1},
            til={Buildings.Types.Tilt.Wall},
            azi={Buildings.Types.Azimuth.E}),
          nConPar=2,
          datConPar(
            layers={conFlo,conFur},
            A={360.0785/hRoo,262.52},
            til={Buildings.Types.Tilt.Floor,Buildings.Types.Tilt.Wall}),
          nConBou=1,
          datConBou(
            layers={conIntWal},
            A={24.13}*hRoo,
            til={Buildings.Types.Tilt.Wall}),
          nSurBou=2,
          surBou(
            each A=6.47*hRoo,
            each absIR=0.9,
            each absSol=0.9,
            til={Buildings.Types.Tilt.Wall, Buildings.Types.Tilt.Wall}),
          nPorts=5,
          intConMod=intConMod,
          energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
          final sampleModel=sampleModel) "East zone"
          annotation (Placement(transformation(extent={{304,56},{344,96}})));
        Buildings.ThermalZones.Detailed.MixedAir nor(
          redeclare package Medium = Medium,
          lat=lat,
          AFlo=568.77/hRoo,
          hRoo=hRoo,
          nConExt=0,
          nConExtWin=1,
          datConExtWin(
            layers={conExtWal},
            A={49.91*hRoo},
            glaSys={glaSys},
            wWin={winWalRat/hWin*49.91*hRoo},
            each hWin=hWin,
            fFra={0.1},
            til={Buildings.Types.Tilt.Wall},
            azi={Buildings.Types.Azimuth.N}),
          nConPar=2,
          datConPar(
            layers={conFlo,conFur},
            A={568.77/hRoo,414.68},
            til={Buildings.Types.Tilt.Floor,Buildings.Types.Tilt.Wall}),
          nConBou=3,
          datConBou(
            layers={conIntWal,conIntWal,conIntWal},
            A={6.47,40.76,6.47}*hRoo,
            til={Buildings.Types.Tilt.Wall, Buildings.Types.Tilt.Wall, Buildings.Types.Tilt.Wall}),
          nSurBou=0,
          nPorts=5,
          intConMod=intConMod,
          energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
          final sampleModel=sampleModel) "North zone"
          annotation (Placement(transformation(extent={{144,116},{184,156}})));
        Buildings.ThermalZones.Detailed.MixedAir wes(
          redeclare package Medium = Medium,
          lat=lat,
          AFlo=360.0785/hRoo,
          hRoo=hRoo,
          nConExt=0,
          nConExtWin=1,
          datConExtWin(
            layers={conExtWal},
            A={33.27*hRoo},
            glaSys={glaSys},
            wWin={winWalRat/hWin*33.27*hRoo},
            each hWin=hWin,
            fFra={0.1},
            til={Buildings.Types.Tilt.Wall},
            azi={Buildings.Types.Azimuth.W}),
          nConPar=2,
          datConPar(
            layers={conFlo,conFur},
            A={360.0785/hRoo,262.52},
            til={Buildings.Types.Tilt.Floor,Buildings.Types.Tilt.Wall}),
          nConBou=1,
          datConBou(
            layers={conIntWal},
            A={24.13}*hRoo,
            til={Buildings.Types.Tilt.Wall}),
          nSurBou=2,
          surBou(
            each A=6.47*hRoo,
            each absIR=0.9,
            each absSol=0.9,
            til={Buildings.Types.Tilt.Wall, Buildings.Types.Tilt.Wall}),
          nPorts=5,
          intConMod=intConMod,
          energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
          final sampleModel=sampleModel) "West zone"
          annotation (Placement(transformation(extent={{12,36},{52,76}})));
        Buildings.ThermalZones.Detailed.MixedAir cor(
          redeclare package Medium = Medium,
          lat=lat,
          AFlo=2698/hRoo,
          hRoo=hRoo,
          nConExt=0,
          nConExtWin=0,
          nConPar=2,
          datConPar(
            layers={conFlo,conFur},
            A={2698/hRoo,1967.01},
            til={Buildings.Types.Tilt.Floor,Buildings.Types.Tilt.Wall}),
          nConBou=0,
          nSurBou=4,
          surBou(
            A={40.76,24.13,40.76,24.13}*hRoo,
            each absIR=0.9,
            each absSol=0.9,
            til={Buildings.Types.Tilt.Wall, Buildings.Types.Tilt.Wall, Buildings.Types.Tilt.Wall, Buildings.Types.Tilt.Wall}),
          nPorts=11,
          intConMod=intConMod,
          energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
          final sampleModel=sampleModel) "Core zone"
          annotation (Placement(transformation(extent={{144,36},{184,76}})));
        Modelica.Fluid.Vessels.BaseClasses.VesselFluidPorts_b portsSou[2](
            redeclare package Medium = Medium) "Fluid inlets and outlets"
          annotation (Placement(transformation(extent={{70,-42},{110,-26}})));
        Modelica.Fluid.Vessels.BaseClasses.VesselFluidPorts_b portsEas[2](
            redeclare package Medium = Medium) "Fluid inlets and outlets"
          annotation (Placement(transformation(extent={{314,28},{354,44}})));
        Modelica.Fluid.Vessels.BaseClasses.VesselFluidPorts_b portsNor[2](
            redeclare package Medium = Medium) "Fluid inlets and outlets"
          annotation (Placement(transformation(extent={{70,118},{110,134}})));
        Modelica.Fluid.Vessels.BaseClasses.VesselFluidPorts_b portsWes[2](
            redeclare package Medium = Medium) "Fluid inlets and outlets"
          annotation (Placement(transformation(extent={{-50,38},{-10,54}})));
        Modelica.Fluid.Vessels.BaseClasses.VesselFluidPorts_b portsCor[2](
            redeclare package Medium = Medium) "Fluid inlets and outlets"
          annotation (Placement(transformation(extent={{70,38},{110,54}})));
        Modelica.Blocks.Math.MatrixGain gai(K=20*[0.4; 0.4; 0.2])
          "Matrix gain to split up heat gain in radiant, convective and latent gain"
          annotation (Placement(transformation(extent={{-100,100},{-80,120}})));
        Modelica.Blocks.Sources.Constant uSha(k=0)
          "Control signal for the shading device"
          annotation (Placement(transformation(extent={{-80,170},{-60,190}})));
        Modelica.Blocks.Routing.Replicator replicator(nout=1)
          annotation (Placement(transformation(extent={{-40,170},{-20,190}})));
        Buildings.BoundaryConditions.WeatherData.Bus weaBus "Weather bus"
          annotation (Placement(transformation(extent={{200,190},{220,210}})));
        RoomLeakage leaSou(redeclare package Medium = Medium, VRoo=568.77,
          s=49.91/33.27,
          azi=Buildings.Types.Azimuth.S,
          final use_windPressure=use_windPressure)
          "Model for air infiltration through the envelope"
          annotation (Placement(transformation(extent={{-58,380},{-22,420}})));
        RoomLeakage leaEas(redeclare package Medium = Medium, VRoo=360.0785,
          s=33.27/49.91,
          azi=Buildings.Types.Azimuth.E,
          final use_windPressure=use_windPressure)
          "Model for air infiltration through the envelope"
          annotation (Placement(transformation(extent={{-58,340},{-22,380}})));
        RoomLeakage leaNor(redeclare package Medium = Medium, VRoo=568.77,
          s=49.91/33.27,
          azi=Buildings.Types.Azimuth.N,
          final use_windPressure=use_windPressure)
          "Model for air infiltration through the envelope"
          annotation (Placement(transformation(extent={{-56,300},{-20,340}})));
        RoomLeakage leaWes(redeclare package Medium = Medium, VRoo=360.0785,
          s=33.27/49.91,
          azi=Buildings.Types.Azimuth.W,
          final use_windPressure=use_windPressure)
          "Model for air infiltration through the envelope"
          annotation (Placement(transformation(extent={{-56,260},{-20,300}})));
        Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temAirSou
          "Air temperature sensor"
          annotation (Placement(transformation(extent={{290,340},{310,360}})));
        Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temAirEas
          "Air temperature sensor"
          annotation (Placement(transformation(extent={{292,310},{312,330}})));
        Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temAirNor
          "Air temperature sensor"
          annotation (Placement(transformation(extent={{292,280},{312,300}})));
        Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temAirWes
          "Air temperature sensor"
          annotation (Placement(transformation(extent={{292,248},{312,268}})));
        Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temAirPer5
          "Air temperature sensor"
          annotation (Placement(transformation(extent={{294,218},{314,238}})));
        Modelica.Blocks.Routing.Multiplex5 multiplex5_1
          annotation (Placement(transformation(extent={{340,280},{360,300}})));
        Modelica.Blocks.Interfaces.RealOutput TRooAir[5](
          each unit="K",
          each displayUnit="degC") "Room air temperatures"
          annotation (Placement(transformation(extent={{380,150},{400,170}}),
              iconTransformation(extent={{380,150},{400,170}})));
        Buildings.Airflow.Multizone.DoorDiscretizedOpen opeSouCor(
          redeclare package Medium = Medium,
          wOpe=10,
          forceErrorControlOnFlow=false) "Opening between perimeter1 and core"
          annotation (Placement(transformation(extent={{84,0},{104,20}})));
        Buildings.Airflow.Multizone.DoorDiscretizedOpen opeEasCor(
          redeclare package Medium = Medium,
          wOpe=10,
          forceErrorControlOnFlow=false) "Opening between perimeter2 and core"
          annotation (Placement(transformation(extent={{250,38},{270,58}})));
        Buildings.Airflow.Multizone.DoorDiscretizedOpen opeNorCor(
          redeclare package Medium = Medium,
          wOpe=10,
          forceErrorControlOnFlow=false) "Opening between perimeter3 and core"
          annotation (Placement(transformation(extent={{80,74},{100,94}})));
        Buildings.Airflow.Multizone.DoorDiscretizedOpen opeWesCor(
          redeclare package Medium = Medium,
          wOpe=10,
          forceErrorControlOnFlow=false) "Opening between perimeter3 and core"
          annotation (Placement(transformation(extent={{20,-20},{40,0}})));
        Modelica.Blocks.Sources.CombiTimeTable intGaiFra(
          table=[0,0.05;
                 8,0.05;
                 9,0.9;
                 12,0.9;
                 12,0.8;
                 13,0.8;
                 13,1;
                 17,1;
                 19,0.1;
                 24,0.05],
          timeScale=3600,
          extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic)
          "Fraction of internal heat gain"
          annotation (Placement(transformation(extent={{-140,100},{-120,120}})));
        Buildings.Fluid.Sensors.RelativePressure senRelPre(redeclare package
            Medium =                                                                  Medium)
          "Building pressure measurement"
          annotation (Placement(transformation(extent={{60,240},{40,260}})));
        Buildings.Fluid.Sources.Outside out(nPorts=1, redeclare package Medium = Medium)
          annotation (Placement(transformation(extent={{-58,240},{-38,260}})));
        Modelica.Blocks.Interfaces.RealOutput p_rel
          "Relative pressure signal of building static pressure" annotation (
            Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=180,
              origin={-170,220})));

        Modelica.Blocks.Math.Gain gaiIntNor[3](each k=kIntNor)
          "Gain for internal heat gain amplification for north zone"
          annotation (Placement(transformation(extent={{-60,134},{-40,154}})));
        Modelica.Blocks.Math.Gain gaiIntSou[3](each k=2 - kIntNor)
          "Gain to change the internal heat gain for south"
          annotation (Placement(transformation(extent={{-60,-38},{-40,-18}})));
      equation
        connect(sou.surf_conBou[1], wes.surf_surBou[2]) annotation (Line(
            points={{170,-40.6667},{170,-54},{62,-54},{62,20},{28.2,20},{28.2,42.5}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(sou.surf_conBou[2], cor.surf_surBou[1]) annotation (Line(
            points={{170,-40},{170,-54},{200,-54},{200,20},{160.2,20},{160.2,41.25}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(sou.surf_conBou[3], eas.surf_surBou[1]) annotation (Line(
            points={{170,-39.3333},{170,-54},{320.2,-54},{320.2,61.5}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(eas.surf_conBou[1], cor.surf_surBou[2]) annotation (Line(
            points={{330,60},{330,20},{160.2,20},{160.2,41.75}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(eas.surf_surBou[2], nor.surf_conBou[1]) annotation (Line(
            points={{320.2,62.5},{320.2,24},{220,24},{220,100},{170,100},{170,119.333}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(nor.surf_conBou[2], cor.surf_surBou[3]) annotation (Line(
            points={{170,120},{170,100},{200,100},{200,26},{160.2,26},{160.2,42.25}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(nor.surf_conBou[3], wes.surf_surBou[1]) annotation (Line(
            points={{170,120.667},{170,100},{60,100},{60,20},{28.2,20},{28.2,41.5}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(wes.surf_conBou[1], cor.surf_surBou[4]) annotation (Line(
            points={{38,40},{38,30},{160.2,30},{160.2,42.75}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(uSha.y, replicator.u) annotation (Line(
            points={{-59,180},{-42,180}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(replicator.y, nor.uSha) annotation (Line(
            points={{-19,180},{130,180},{130,154},{142.4,154}},
            color={0,0,127},
            pattern=LinePattern.Dash,
            smooth=Smooth.None));
        connect(replicator.y, wes.uSha) annotation (Line(
            points={{-19,180},{-6,180},{-6,74},{10.4,74}},
            color={0,0,127},
            pattern=LinePattern.Dash,
            smooth=Smooth.None));
        connect(replicator.y, eas.uSha) annotation (Line(
            points={{-19,180},{232,180},{232,94},{302.4,94}},
            color={0,0,127},
            pattern=LinePattern.Dash,
            smooth=Smooth.None));
        connect(replicator.y, sou.uSha) annotation (Line(
            points={{-19,180},{130,180},{130,-6},{142.4,-6}},
            color={0,0,127},
            pattern=LinePattern.Dash,
            smooth=Smooth.None));
        connect(replicator.y, cor.uSha) annotation (Line(
            points={{-19,180},{130,180},{130,74},{142.4,74}},
            color={0,0,127},
            pattern=LinePattern.Dash,
            smooth=Smooth.None));
        connect(gai.y, cor.qGai_flow)          annotation (Line(
            points={{-79,110},{120,110},{120,64},{142.4,64}},
            color={0,0,127},
            pattern=LinePattern.Dash,
            smooth=Smooth.None));
        connect(gai.y, eas.qGai_flow)          annotation (Line(
            points={{-79,110},{226,110},{226,84},{302.4,84}},
            color={0,0,127},
            pattern=LinePattern.Dash,
            smooth=Smooth.None));
        connect(gai.y, wes.qGai_flow)          annotation (Line(
            points={{-79,110},{-14,110},{-14,64},{10.4,64}},
            color={0,0,127},
            pattern=LinePattern.Dash,
            smooth=Smooth.None));
        connect(sou.weaBus, weaBus) annotation (Line(
            points={{181.9,-6.1},{181.9,8},{210,8},{210,200}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%second",
            index=1,
            extent={{6,3},{6,3}}));
        connect(eas.weaBus, weaBus) annotation (Line(
            points={{341.9,93.9},{341.9,120},{210,120},{210,200}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(nor.weaBus, weaBus) annotation (Line(
            points={{181.9,153.9},{182,160},{182,168},{210,168},{210,200}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(wes.weaBus, weaBus) annotation (Line(
            points={{49.9,73.9},{49.9,168},{210,168},{210,200}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(cor.weaBus, weaBus) annotation (Line(
            points={{181.9,73.9},{181.9,90},{210,90},{210,200}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(weaBus, leaSou.weaBus) annotation (Line(
            points={{210,200},{-80,200},{-80,400},{-58,400}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(weaBus, leaEas.weaBus) annotation (Line(
            points={{210,200},{-80,200},{-80,360},{-58,360}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(weaBus, leaNor.weaBus) annotation (Line(
            points={{210,200},{-80,200},{-80,320},{-56,320}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(weaBus, leaWes.weaBus) annotation (Line(
            points={{210,200},{-80,200},{-80,280},{-56,280}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(multiplex5_1.y, TRooAir) annotation (Line(
            points={{361,290},{372,290},{372,160},{390,160}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(temAirSou.T, multiplex5_1.u1[1]) annotation (Line(
            points={{310,350},{328,350},{328,300},{338,300}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(temAirEas.T, multiplex5_1.u2[1]) annotation (Line(
            points={{312,320},{324,320},{324,295},{338,295}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(temAirNor.T, multiplex5_1.u3[1]) annotation (Line(
            points={{312,290},{338,290}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(temAirWes.T, multiplex5_1.u4[1]) annotation (Line(
            points={{312,258},{324,258},{324,285},{338,285}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(temAirPer5.T, multiplex5_1.u5[1]) annotation (Line(
            points={{314,228},{322,228},{322,228},{332,228},{332,280},{338,280}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(sou.heaPorAir, temAirSou.port) annotation (Line(
            points={{163,-24},{224,-24},{224,100},{264,100},{264,350},{290,350}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(eas.heaPorAir, temAirEas.port) annotation (Line(
            points={{323,76},{286,76},{286,320},{292,320}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(nor.heaPorAir, temAirNor.port) annotation (Line(
            points={{163,136},{164,136},{164,290},{292,290}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(wes.heaPorAir, temAirWes.port) annotation (Line(
            points={{31,56},{70,56},{70,114},{186,114},{186,258},{292,258}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(cor.heaPorAir, temAirPer5.port) annotation (Line(
            points={{163,56},{162,56},{162,228},{294,228}},
            color={191,0,0},
            smooth=Smooth.None));
        connect(sou.ports[1], portsSou[1]) annotation (Line(
            points={{149,-37.2},{114,-37.2},{114,-34},{80,-34}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(sou.ports[2], portsSou[2]) annotation (Line(
            points={{149,-35.6},{124,-35.6},{124,-34},{100,-34}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(eas.ports[1], portsEas[1]) annotation (Line(
            points={{309,62.8},{300,62.8},{300,36},{324,36}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(eas.ports[2], portsEas[2]) annotation (Line(
            points={{309,64.4},{300,64.4},{300,36},{344,36}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(nor.ports[1], portsNor[1]) annotation (Line(
            points={{149,122.8},{114,122.8},{114,126},{80,126}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(nor.ports[2], portsNor[2]) annotation (Line(
            points={{149,124.4},{124,124.4},{124,126},{100,126}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(wes.ports[1], portsWes[1]) annotation (Line(
            points={{17,42.8},{-12,42.8},{-12,46},{-40,46}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(wes.ports[2], portsWes[2]) annotation (Line(
            points={{17,44.4},{-2,44.4},{-2,46},{-20,46}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(cor.ports[1], portsCor[1]) annotation (Line(
            points={{149,42.3636},{114,42.3636},{114,46},{80,46}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(cor.ports[2], portsCor[2]) annotation (Line(
            points={{149,43.0909},{124,43.0909},{124,46},{100,46}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(leaSou.port_b, sou.ports[3]) annotation (Line(
            points={{-22,400},{-2,400},{-2,-72},{134,-72},{134,-34},{149,-34}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(leaEas.port_b, eas.ports[3]) annotation (Line(
            points={{-22,360},{246,360},{246,66},{309,66}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(leaNor.port_b, nor.ports[3]) annotation (Line(
            points={{-20,320},{138,320},{138,126},{149,126}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(leaWes.port_b, wes.ports[3]) annotation (Line(
            points={{-20,280},{2,280},{2,46},{17,46}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeSouCor.port_b1, cor.ports[3]) annotation (Line(
            points={{104,16},{116,16},{116,43.8182},{149,43.8182}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeSouCor.port_a2, cor.ports[4]) annotation (Line(
            points={{104,4},{116,4},{116,44.5455},{149,44.5455}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeSouCor.port_a1, sou.ports[4]) annotation (Line(
            points={{84,16},{74,16},{74,-20},{134,-20},{134,-32.4},{149,-32.4}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeSouCor.port_b2, sou.ports[5]) annotation (Line(
            points={{84,4},{74,4},{74,-20},{134,-20},{134,-30.8},{149,-30.8}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeEasCor.port_b1, eas.ports[4]) annotation (Line(
            points={{270,54},{290,54},{290,67.6},{309,67.6}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeEasCor.port_a2, eas.ports[5]) annotation (Line(
            points={{270,42},{290,42},{290,69.2},{309,69.2}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeEasCor.port_a1, cor.ports[5]) annotation (Line(
            points={{250,54},{190,54},{190,34},{142,34},{142,45.2727},{149,45.2727}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeEasCor.port_b2, cor.ports[6]) annotation (Line(
            points={{250,42},{190,42},{190,34},{142,34},{142,46},{149,46}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeNorCor.port_b1, nor.ports[4]) annotation (Line(
            points={{100,90},{124,90},{124,127.6},{149,127.6}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeNorCor.port_a2, nor.ports[5]) annotation (Line(
            points={{100,78},{124,78},{124,129.2},{149,129.2}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeNorCor.port_a1, cor.ports[7]) annotation (Line(
            points={{80,90},{76,90},{76,60},{142,60},{142,46.7273},{149,46.7273}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(opeNorCor.port_b2, cor.ports[8]) annotation (Line(
            points={{80,78},{76,78},{76,60},{142,60},{142,47.4545},{149,47.4545}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeWesCor.port_b1, cor.ports[9]) annotation (Line(
            points={{40,-4},{56,-4},{56,34},{116,34},{116,48.1818},{149,48.1818}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeWesCor.port_a2, cor.ports[10]) annotation (Line(
            points={{40,-16},{56,-16},{56,34},{116,34},{116,48.9091},{149,48.9091}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeWesCor.port_a1, wes.ports[4]) annotation (Line(
            points={{20,-4},{2,-4},{2,47.6},{17,47.6}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(opeWesCor.port_b2, wes.ports[5]) annotation (Line(
            points={{20,-16},{2,-16},{2,49.2},{17,49.2}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(intGaiFra.y, gai.u) annotation (Line(
            points={{-119,110},{-102,110}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(cor.ports[11], senRelPre.port_a) annotation (Line(
            points={{149,49.6364},{110,49.6364},{110,250},{60,250}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(out.weaBus, weaBus) annotation (Line(
            points={{-58,250.2},{-70,250.2},{-70,250},{-80,250},{-80,200},{210,200}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None), Text(
            textString="%second",
            index=1,
            extent={{6,3},{6,3}}));
        connect(out.ports[1], senRelPre.port_b) annotation (Line(
            points={{-38,250},{40,250}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(senRelPre.p_rel, p_rel) annotation (Line(
            points={{50,241},{50,220},{-170,220}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(gai.y, gaiIntNor.u) annotation (Line(
            points={{-79,110},{-68,110},{-68,144},{-62,144}},
            color={0,0,127},
            pattern=LinePattern.Dash));
        connect(gaiIntNor.y, nor.qGai_flow) annotation (Line(
            points={{-39,144},{142.4,144}},
            color={0,0,127},
            pattern=LinePattern.Dash));
        connect(gai.y, gaiIntSou.u) annotation (Line(
            points={{-79,110},{-68,110},{-68,-28},{-62,-28}},
            color={0,0,127},
            pattern=LinePattern.Dash));
        connect(gaiIntSou.y, sou.qGai_flow) annotation (Line(
            points={{-39,-28},{68,-28},{68,-16},{142.4,-16}},
            color={0,0,127},
            pattern=LinePattern.Dash));
        annotation (Diagram(coordinateSystem(preserveAspectRatio=true, extent={{-160,-100},
                  {400,500}},
              initialScale=0.1)),     Icon(coordinateSystem(
                preserveAspectRatio=true, extent={{-160,-100},{400,500}}), graphics={
              Rectangle(
                extent={{-80,-80},{380,180}},
                lineColor={95,95,95},
                fillColor={95,95,95},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{-60,160},{360,-60}},
                pattern=LinePattern.None,
                lineColor={117,148,176},
                fillColor={170,213,255},
                fillPattern=FillPattern.Sphere),
              Rectangle(
                extent={{0,-80},{294,-60}},
                lineColor={95,95,95},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{0,-74},{294,-66}},
                lineColor={95,95,95},
                fillColor={170,213,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{8,8},{294,100}},
                lineColor={95,95,95},
                fillColor={95,95,95},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{20,88},{280,22}},
                pattern=LinePattern.None,
                lineColor={117,148,176},
                fillColor={170,213,255},
                fillPattern=FillPattern.Sphere),
              Polygon(
                points={{-56,170},{20,94},{12,88},{-62,162},{-56,170}},
                smooth=Smooth.None,
                fillColor={95,95,95},
                fillPattern=FillPattern.Solid,
                pattern=LinePattern.None),
              Polygon(
                points={{290,16},{366,-60},{358,-66},{284,8},{290,16}},
                smooth=Smooth.None,
                fillColor={95,95,95},
                fillPattern=FillPattern.Solid,
                pattern=LinePattern.None),
              Polygon(
                points={{284,96},{360,168},{368,162},{292,90},{284,96}},
                smooth=Smooth.None,
                fillColor={95,95,95},
                fillPattern=FillPattern.Solid,
                pattern=LinePattern.None),
              Rectangle(
                extent={{-80,120},{-60,-20}},
                lineColor={95,95,95},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{-74,120},{-66,-20}},
                lineColor={95,95,95},
                fillColor={170,213,255},
                fillPattern=FillPattern.Solid),
              Polygon(
                points={{-64,-56},{18,22},{26,16},{-58,-64},{-64,-56}},
                smooth=Smooth.None,
                fillColor={95,95,95},
                fillPattern=FillPattern.Solid,
                pattern=LinePattern.None),
              Rectangle(
                extent={{360,122},{380,-18}},
                lineColor={95,95,95},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{366,122},{374,-18}},
                lineColor={95,95,95},
                fillColor={170,213,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{2,170},{296,178}},
                lineColor={95,95,95},
                fillColor={170,213,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{2,160},{296,180}},
                lineColor={95,95,95},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{2,166},{296,174}},
                lineColor={95,95,95},
                fillColor={170,213,255},
                fillPattern=FillPattern.Solid),
              Text(
                extent={{-84,234},{-62,200}},
                lineColor={0,0,255},
                textString="dP")}),
          Documentation(revisions="<html>
<ul>
<li>
May 1, 2013, by Michael Wetter:<br/>
Declared the parameter record to be a parameter, as declaring its elements
to be parameters does not imply that the whole record has the variability of a parameter.
</li>
<li>
January 23, 2020, by Milica Grahovac:<br/>
Updated core zone geometry parameters related to 
room heat and mass balance.
</li>
</ul>
</html>",       info="<html>
<p>
Model of a floor that consists
of five thermal zones that are representative of one floor of the
new construction medium office building for Chicago, IL,
as described in the set of DOE Commercial Building Benchmarks.
There are four perimeter zones and one core zone.
The envelope thermal properties meet ASHRAE Standard 90.1-2004.
</p>
</html>"));
      end Floor;

      model RoomLeakage "Room leakage model"
        extends Buildings.BaseClasses.BaseIcon;
        replaceable package Medium = Modelica.Media.Interfaces.PartialMedium
          "Medium in the component" annotation (choicesAllMatching=true);
        parameter Modelica.SIunits.Volume VRoo "Room volume";
        parameter Boolean use_windPressure=false
          "Set to true to enable wind pressure"
          annotation(Evaluate=true);
        Buildings.Fluid.FixedResistances.PressureDrop res(
          redeclare package Medium = Medium,
          dp_nominal=50,
          m_flow_nominal=VRoo*1.2/3600) "Resistance model"
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));
        Modelica.Fluid.Interfaces.FluidPort_b port_b(redeclare package Medium =
              Medium) annotation (Placement(transformation(extent={{90,-10},{110,10}})));
        Buildings.Fluid.Sources.Outside_CpLowRise
                              amb(redeclare package Medium = Medium, nPorts=1,
          s=s,
          azi=azi,
          Cp0=if use_windPressure then 0.6 else 0)
          annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
        Buildings.BoundaryConditions.WeatherData.Bus weaBus "Bus with weather data"
          annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
        Buildings.Fluid.Sensors.MassFlowRate senMasFlo1(redeclare package
            Medium =                                                               Medium,
            allowFlowReversal=true) "Sensor for mass flow rate" annotation (Placement(
              transformation(
              extent={{10,10},{-10,-10}},
              rotation=180,
              origin={-10,0})));
        Modelica.Blocks.Math.Gain ACHInf(k=1/VRoo/1.2*3600, y(unit="1/h"))
          "Air change per hour due to infiltration"
          annotation (Placement(transformation(extent={{12,30},{32,50}})));
        parameter Real s "Side ratio, s=length of this wall/length of adjacent wall";
        parameter Modelica.SIunits.Angle azi "Surface azimuth (South:0, West:pi/2)";

      equation
        connect(res.port_b, port_b) annotation (Line(points={{40,6.10623e-16},{55,
                6.10623e-16},{55,1.16573e-15},{70,1.16573e-15},{70,5.55112e-16},{100,
                5.55112e-16}},                                   color={0,127,255}));
        connect(amb.weaBus, weaBus) annotation (Line(
            points={{-60,0.2},{-80,0.2},{-80,5.55112e-16},{-100,5.55112e-16}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(amb.ports[1], senMasFlo1.port_a) annotation (Line(
            points={{-40,6.66134e-16},{-20,6.66134e-16},{-20,7.25006e-16}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(senMasFlo1.port_b, res.port_a) annotation (Line(
            points={{5.55112e-16,-1.72421e-15},{10,-1.72421e-15},{10,6.10623e-16},{20,
                6.10623e-16}},
            color={0,127,255},
            smooth=Smooth.None));
        connect(senMasFlo1.m_flow, ACHInf.u) annotation (Line(
            points={{-10,11},{-10,40},{10,40}},
            color={0,0,127},
            smooth=Smooth.None));
        annotation (
          Icon(coordinateSystem(preserveAspectRatio=false, extent={{-100,-100},{100,
                  100}}), graphics={
              Ellipse(
                extent={{-80,40},{0,-40}},
                lineColor={0,0,0},
                fillPattern=FillPattern.Sphere,
                fillColor={0,127,255}),
              Rectangle(
                extent={{20,12},{80,-12}},
                lineColor={0,0,0},
                fillPattern=FillPattern.HorizontalCylinder,
                fillColor={192,192,192}),
              Rectangle(
                extent={{20,6},{80,-6}},
                lineColor={0,0,0},
                fillPattern=FillPattern.HorizontalCylinder,
                fillColor={0,127,255}),
              Line(points={{-100,0},{-80,0}}, color={0,0,255}),
              Line(points={{0,0},{20,0}}, color={0,0,255}),
              Line(points={{80,0},{90,0}}, color={0,0,255})}),
          Documentation(info="<html>
<p>
Room leakage.
</p></html>",       revisions="<html>
<ul>
<li>
July 20, 2007 by Michael Wetter:<br/>
First implementation.
</li>
</ul>
</html>"));
      end RoomLeakage;

      model VAVBranch "Supply branch of a VAV system"
        extends Modelica.Blocks.Icons.Block;
        replaceable package MediumA = Modelica.Media.Interfaces.PartialMedium
          "Medium model for air" annotation (choicesAllMatching=true);
        replaceable package MediumW = Modelica.Media.Interfaces.PartialMedium
          "Medium model for water" annotation (choicesAllMatching=true);

        parameter Boolean allowFlowReversal=true
          "= false to simplify equations, assuming, but not enforcing, no flow reversal";

        parameter Modelica.SIunits.MassFlowRate m_flow_nominal
          "Mass flow rate of this thermal zone";
        parameter Modelica.SIunits.Volume VRoo "Room volume";

        Buildings.Fluid.Actuators.Dampers.PressureIndependent vav(
          redeclare package Medium = MediumA,
          m_flow_nominal=m_flow_nominal,
          dpDamper_nominal=220 + 20,
          allowFlowReversal=allowFlowReversal) "VAV box for room" annotation (
            Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={-50,40})));
        Buildings.Fluid.HeatExchangers.DryCoilEffectivenessNTU terHea(
          redeclare package Medium1 = MediumA,
          redeclare package Medium2 = MediumW,
          m1_flow_nominal=m_flow_nominal,
          m2_flow_nominal=m_flow_nominal*1000*(50 - 17)/4200/10,
          Q_flow_nominal=m_flow_nominal*1006*(50 - 16.7),
          configuration=Buildings.Fluid.Types.HeatExchangerConfiguration.CounterFlow,
          dp1_nominal=0,
          from_dp2=true,
          dp2_nominal=0,
          allowFlowReversal1=allowFlowReversal,
          allowFlowReversal2=false,
          T_a1_nominal=289.85,
          T_a2_nominal=355.35) "Heat exchanger of terminal box" annotation (Placement(
              transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={-44,0})));
        Buildings.Fluid.Sources.Boundary_pT sinTer(
          redeclare package Medium = MediumW,
          p(displayUnit="Pa") = 3E5,
          nPorts=1) "Sink for terminal box " annotation (Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=180,
              origin={40,-20})));
        Modelica.Fluid.Interfaces.FluidPort_a port_a(
          redeclare package Medium = MediumA)
          "Fluid connector a1 (positive design flow direction is from port_a1 to port_b1)"
          annotation (Placement(transformation(extent={{-60,-110},{-40,-90}}),
              iconTransformation(extent={{-60,-110},{-40,-90}})));
        Modelica.Fluid.Interfaces.FluidPort_a port_b(
          redeclare package Medium = MediumA)
          "Fluid connector b (positive design flow direction is from port_a1 to port_b1)"
          annotation (Placement(transformation(extent={{-60,90},{-40,110}}),
              iconTransformation(extent={{-60,90},{-40,110}})));
        Buildings.Fluid.Sensors.MassFlowRate senMasFlo(
          redeclare package Medium = MediumA,
          allowFlowReversal=allowFlowReversal)
          "Sensor for mass flow rate" annotation (Placement(
              transformation(
              extent={{-10,10},{10,-10}},
              rotation=90,
              origin={-50,70})));
        Modelica.Blocks.Math.Gain fraMasFlo(k=1/m_flow_nominal)
          "Fraction of mass flow rate, relative to nominal flow"
          annotation (Placement(transformation(extent={{0,70},{20,90}})));
        Modelica.Blocks.Math.Gain ACH(k=1/VRoo/1.2*3600) "Air change per hour"
          annotation (Placement(transformation(extent={{0,30},{20,50}})));
        Buildings.Fluid.Sources.MassFlowSource_T souTer(
          redeclare package Medium = MediumW,
          nPorts=1,
          use_m_flow_in=true,
          T=323.15) "Source for terminal box " annotation (Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=180,
              origin={40,20})));
        Modelica.Blocks.Interfaces.RealInput yVAV "Signal for VAV damper"
                                                                  annotation (
            Placement(transformation(extent={{-140,20},{-100,60}}),
              iconTransformation(extent={{-140,20},{-100,60}})));
        Modelica.Blocks.Interfaces.RealInput yVal
          "Actuator position for reheat valve (0: closed, 1: open)" annotation (
            Placement(transformation(extent={{-140,-60},{-100,-20}}),
              iconTransformation(extent={{-140,-60},{-100,-20}})));
        Buildings.Controls.OBC.CDL.Continuous.Gain gaiM_flow(
          final k=m_flow_nominal*1000*15/4200/10) "Gain for mass flow rate"
          annotation (Placement(transformation(extent={{80,2},{60,22}})));
        Modelica.Blocks.Interfaces.RealOutput y_actual "Actual VAV damper position"
          annotation (Placement(transformation(extent={{100,46},{120,66}}),
              iconTransformation(extent={{100,70},{120,90}})));
      equation
        connect(fraMasFlo.u, senMasFlo.m_flow) annotation (Line(
            points={{-2,80},{-24,80},{-24,70},{-39,70}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(vav.port_b, senMasFlo.port_a) annotation (Line(
            points={{-50,50},{-50,60}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(ACH.u, senMasFlo.m_flow) annotation (Line(
            points={{-2,40},{-24,40},{-24,70},{-39,70}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(souTer.ports[1], terHea.port_a2) annotation (Line(
            points={{30,20},{-38,20},{-38,10}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(port_a, terHea.port_a1) annotation (Line(
            points={{-50,-100},{-50,-10}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(senMasFlo.port_b, port_b) annotation (Line(
            points={{-50,80},{-50,100}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(terHea.port_b1, vav.port_a) annotation (Line(
            points={{-50,10},{-50,30}},
            color={0,127,255},
            thickness=0.5));
        connect(vav.y, yVAV) annotation (Line(points={{-62,40},{-120,40}},
                      color={0,0,127}));
        connect(souTer.m_flow_in, gaiM_flow.y)
          annotation (Line(points={{52,12},{58,12}}, color={0,0,127}));
        connect(sinTer.ports[1], terHea.port_b2) annotation (Line(points={{30,-20},{
                -38,-20},{-38,-10}}, color={0,127,255}));
        connect(gaiM_flow.u, yVal) annotation (Line(points={{82,12},{90,12},{90,-40},
                {-120,-40}}, color={0,0,127}));
        connect(vav.y_actual, y_actual)
          annotation (Line(points={{-57,45},{-57,56},{110,56}}, color={0,0,127}));
        annotation (Icon(
          graphics={
              Rectangle(
                extent={{-108.07,-16.1286},{93.93,-20.1286}},
                lineColor={0,0,0},
                fillPattern=FillPattern.HorizontalCylinder,
                fillColor={0,127,255},
                origin={-68.1286,6.07},
                rotation=90),
              Rectangle(
                extent={{-68,-20},{-26,-60}},
                fillPattern=FillPattern.Solid,
                fillColor={175,175,175},
                pattern=LinePattern.None),
              Rectangle(
                extent={{100.8,-22},{128.8,-44}},
                lineColor={0,0,0},
                fillPattern=FillPattern.HorizontalCylinder,
                fillColor={192,192,192},
                origin={-82,-76.8},
                rotation=90),
              Rectangle(
                extent={{102.2,-11.6667},{130.2,-25.6667}},
                lineColor={0,0,0},
                fillPattern=FillPattern.HorizontalCylinder,
                fillColor={0,127,255},
                origin={-67.6667,-78.2},
                rotation=90),
              Polygon(
                points={{-62,32},{-34,48},{-34,46},{-62,30},{-62,32}},
                pattern=LinePattern.None,
                smooth=Smooth.None,
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid,
                lineColor={0,0,0}),
              Polygon(
                points={{-68,-28},{-34,-28},{-34,-30},{-68,-30},{-68,-28}},
                pattern=LinePattern.None,
                smooth=Smooth.None,
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid,
                lineColor={0,0,0}),
              Polygon(
                points={{-68,-52},{-34,-52},{-34,-54},{-68,-54},{-68,-52}},
                pattern=LinePattern.None,
                smooth=Smooth.None,
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid,
                lineColor={0,0,0}),
              Polygon(
                points={{-48,-34},{-34,-28},{-34,-30},{-48,-36},{-48,-34}},
                pattern=LinePattern.None,
                smooth=Smooth.None,
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid,
                lineColor={0,0,0}),
              Polygon(
                points={{-48,-34},{-34,-40},{-34,-42},{-48,-36},{-48,-34}},
                pattern=LinePattern.None,
                smooth=Smooth.None,
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid,
                lineColor={0,0,0}),
              Polygon(
                points={{-48,-46},{-34,-52},{-34,-54},{-48,-48},{-48,-46}},
                pattern=LinePattern.None,
                smooth=Smooth.None,
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid,
                lineColor={0,0,0}),
              Polygon(
                points={{-48,-46},{-34,-40},{-34,-42},{-48,-48},{-48,-46}},
                pattern=LinePattern.None,
                smooth=Smooth.None,
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid,
                lineColor={0,0,0})}), Documentation(info="<html>
<p>
Model for a VAV supply branch. 
The terminal VAV box has a pressure independent damper and a water reheat coil. 
The pressure independent damper model includes an idealized flow rate controller 
and requires a discharge air flow rate set-point (normalized to the nominal value) 
as a control signal.
</p>
</html>"));
      end VAVBranch;
    end ThermalZones;

    package BaseClasses "Package with base classes for Buildings.Examples.VAVReheat"
    extends Modelica.Icons.BasesPackage;

      model MixingBox
        "Outside air mixing box with non-interlocked air dampers"

        replaceable package Medium =
            Modelica.Media.Interfaces.PartialMedium "Medium in the component"
          annotation (choicesAllMatching = true);
        import Modelica.Constants;

        parameter Boolean allowFlowReversal = true
          "= false to simplify equations, assuming, but not enforcing, no flow reversal"
          annotation(Dialog(tab="Assumptions"), Evaluate=true);

        parameter Boolean use_deltaM = true
          "Set to true to use deltaM for turbulent transition, else ReC is used";
        parameter Real deltaM = 0.3
          "Fraction of nominal mass flow rate where transition to turbulent occurs"
          annotation(Dialog(enable=use_deltaM));
        parameter Modelica.SIunits.Velocity v_nominal=1 "Nominal face velocity";

        parameter Boolean roundDuct = false
          "Set to true for round duct, false for square cross section"
          annotation(Dialog(enable=not use_deltaM));
        parameter Real ReC=4000
          "Reynolds number where transition to turbulent starts"
          annotation(Dialog(enable=not use_deltaM));

        parameter Boolean dp_nominalIncludesDamper=false
          "set to true if dp_nominal includes the pressure loss of the open damper"
          annotation (Dialog(group="Nominal condition"));

        parameter Modelica.SIunits.MassFlowRate mOut_flow_nominal
          "Mass flow rate outside air damper"
          annotation (Dialog(group="Nominal condition"));
        parameter Modelica.SIunits.PressureDifference dpOut_nominal(min=0, displayUnit="Pa")
          "Pressure drop outside air leg"
           annotation (Dialog(group="Nominal condition"));

        parameter Modelica.SIunits.MassFlowRate mRec_flow_nominal
          "Mass flow rate recirculation air damper"
          annotation (Dialog(group="Nominal condition"));
        parameter Modelica.SIunits.PressureDifference dpRec_nominal(min=0, displayUnit="Pa")
          "Pressure drop recirculation air leg"
           annotation (Dialog(group="Nominal condition"));

        parameter Modelica.SIunits.MassFlowRate mExh_flow_nominal
          "Mass flow rate exhaust air damper"
          annotation (Dialog(group="Nominal condition"));
        parameter Modelica.SIunits.PressureDifference dpExh_nominal(min=0, displayUnit="Pa")
          "Pressure drop exhaust air leg"
           annotation (Dialog(group="Nominal condition"));

        parameter Boolean from_dp=true
          "= true, use m_flow = f(dp) else dp = f(m_flow)"
          annotation (Dialog(tab="Advanced"));
        parameter Boolean linearized=false
          "= true, use linear relation between m_flow and dp for any flow rate"
          annotation (Dialog(tab="Advanced"));
        parameter Boolean use_constant_density=true
          "Set to true to use constant density for flow friction"
          annotation (Dialog(tab="Advanced"));
        parameter Real a=-1.51 "Coefficient a for damper characteristics"
          annotation (Dialog(tab="Damper coefficients"));
        parameter Real b=0.105*90 "Coefficient b for damper characteristics"
          annotation (Dialog(tab="Damper coefficients"));
        parameter Real yL=15/90 "Lower value for damper curve"
          annotation (Dialog(tab="Damper coefficients"));
        parameter Real yU=55/90 "Upper value for damper curve"
          annotation (Dialog(tab="Damper coefficients"));
        parameter Real k1=0.45
          "Flow coefficient for y=1, k1 = pressure drop divided by dynamic pressure"
          annotation (Dialog(tab="Damper coefficients"));

        parameter Modelica.SIunits.Time riseTime=15
          "Rise time of the filter (time to reach 99.6 % of an opening step)"
          annotation (Dialog(tab="Dynamics", group="Filtered opening"));
        parameter Modelica.Blocks.Types.Init init=Modelica.Blocks.Types.Init.InitialOutput
          "Type of initialization (no init/steady state/initial state/initial output)"
          annotation (Dialog(tab="Dynamics", group="Filtered opening"));
        parameter Real y_start=1 "Initial value of output"
          annotation (Dialog(tab="Dynamics", group="Filtered opening"));

        Modelica.Blocks.Interfaces.RealInput yRet(
          min=0,
          max=1,
          final unit="1")
          "Return damper position (0: closed, 1: open)" annotation (Placement(
              transformation(
              extent={{-20,-20},{20,20}},
              rotation=270,
              origin={-68,120}), iconTransformation(
              extent={{-20,-20},{20,20}},
              rotation=270,
              origin={-68,120})));
        Modelica.Blocks.Interfaces.RealInput yOut(
          min=0,
          max=1,
          final unit="1")
          "Outdoor air damper signal (0: closed, 1: open)" annotation (Placement(
              transformation(
              extent={{-20,-20},{20,20}},
              rotation=270,
              origin={0,120}), iconTransformation(
              extent={{-20,-20},{20,20}},
              rotation=270,
              origin={0,120})));
        Modelica.Blocks.Interfaces.RealInput yExh(
          min=0,
          max=1,
          final unit="1")
          "Exhaust air damper signal (0: closed, 1: open)" annotation (Placement(
              transformation(
              extent={{-20,-20},{20,20}},
              rotation=270,
              origin={60,120}), iconTransformation(
              extent={{-20,-20},{20,20}},
              rotation=270,
              origin={70,120})));

        Modelica.Fluid.Interfaces.FluidPort_a port_Out(redeclare package Medium =
              Medium, m_flow(start=0, min=if allowFlowReversal then -Constants.inf else
                      0))
          "Fluid connector a (positive design flow direction is from port_a to port_b)"
          annotation (Placement(transformation(extent={{-110,50},{-90,70}})));
        Modelica.Fluid.Interfaces.FluidPort_b port_Exh(redeclare package Medium =
              Medium, m_flow(start=0, max=if allowFlowReversal then +Constants.inf else
                      0))
          "Fluid connector b (positive design flow direction is from port_a to port_b)"
          annotation (Placement(transformation(extent={{-90,-70},{-110,-50}})));
        Modelica.Fluid.Interfaces.FluidPort_a port_Ret(redeclare package Medium =
              Medium, m_flow(start=0, min=if allowFlowReversal then -Constants.inf else
                      0))
          "Fluid connector a (positive design flow direction is from port_a to port_b)"
          annotation (Placement(transformation(extent={{110,-70},{90,-50}})));
        Modelica.Fluid.Interfaces.FluidPort_b port_Sup(redeclare package Medium =
              Medium, m_flow(start=0, max=if allowFlowReversal then +Constants.inf else
                      0))
          "Fluid connector b (positive design flow direction is from port_a to port_b)"
          annotation (Placement(transformation(extent={{110,50},{90,70}})));

        Buildings.Fluid.Actuators.Dampers.Exponential damOut(
          redeclare package Medium = Medium,
          from_dp=from_dp,
          linearized=linearized,
          use_deltaM=use_deltaM,
          deltaM=deltaM,
          roundDuct=roundDuct,
          ReC=ReC,
          a=a,
          b=b,
          yL=yL,
          yU=yU,
          use_constant_density=use_constant_density,
          allowFlowReversal=allowFlowReversal,
          m_flow_nominal=mOut_flow_nominal,
          use_inputFilter=true,
          final riseTime=riseTime,
          final init=init,
          y_start=y_start,
          dpDamper_nominal=(k1)*1.2*(1)^2/2,
          dpFixed_nominal=if (dp_nominalIncludesDamper) then (dpOut_nominal) - (k1)*
              1.2*(1)^2/2 else (dpOut_nominal),
          k1=k1) "Outdoor air damper"
          annotation (Placement(transformation(extent={{-10,50},{10,70}})));

        Buildings.Fluid.Actuators.Dampers.Exponential damExh(
          redeclare package Medium = Medium,
          m_flow_nominal=mExh_flow_nominal,
          from_dp=from_dp,
          linearized=linearized,
          use_deltaM=use_deltaM,
          deltaM=deltaM,
          roundDuct=roundDuct,
          ReC=ReC,
          a=a,
          b=b,
          yL=yL,
          yU=yU,
          use_constant_density=use_constant_density,
          allowFlowReversal=allowFlowReversal,
          use_inputFilter=true,
          final riseTime=riseTime,
          final init=init,
          y_start=y_start,
          dpDamper_nominal=(k1)*1.2*(1)^2/2,
          dpFixed_nominal=if (dp_nominalIncludesDamper) then (dpExh_nominal) - (k1)*
              1.2*(1)^2/2 else (dpExh_nominal),
          k1=k1) "Exhaust air damper"
          annotation (Placement(transformation(extent={{-20,-70},{-40,-50}})));

        Buildings.Fluid.Actuators.Dampers.Exponential damRet(
          redeclare package Medium = Medium,
          m_flow_nominal=mRec_flow_nominal,
          from_dp=from_dp,
          linearized=linearized,
          use_deltaM=use_deltaM,
          deltaM=deltaM,
          roundDuct=roundDuct,
          ReC=ReC,
          a=a,
          b=b,
          yL=yL,
          yU=yU,
          use_constant_density=use_constant_density,
          allowFlowReversal=allowFlowReversal,
          use_inputFilter=true,
          final riseTime=riseTime,
          final init=init,
          y_start=y_start,
          dpDamper_nominal=(k1)*1.2*(1)^2/2,
          dpFixed_nominal=if (dp_nominalIncludesDamper) then (dpRec_nominal) - (k1)*
              1.2*(1)^2/2 else (dpRec_nominal),
          k1=k1) "Return air damper" annotation (Placement(transformation(
              origin={80,0},
              extent={{-10,-10},{10,10}},
              rotation=90)));

      protected
        parameter Medium.Density rho_default=Medium.density(sta_default)
          "Density, used to compute fluid volume";
        parameter Medium.ThermodynamicState sta_default=
           Medium.setState_pTX(T=Medium.T_default, p=Medium.p_default, X=Medium.X_default)
          "Default medium state";

      equation
        connect(damOut.port_a, port_Out)
          annotation (Line(points={{-10,60},{-100,60}}, color={0,127,255}));
        connect(damExh.port_b, port_Exh) annotation (Line(
            points={{-40,-60},{-100,-60}},
            color={0,127,255}));
        connect(port_Sup, damOut.port_b)
          annotation (Line(points={{100,60},{10,60}}, color={0,127,255}));
        connect(damRet.port_b, port_Sup) annotation (Line(
            points={{80,10},{80,60},{100,60}},
            color={0,127,255}));
        connect(port_Ret, damExh.port_a) annotation (Line(
            points={{100,-60},{-20,-60}},
            color={0,127,255}));
        connect(port_Ret,damRet. port_a) annotation (Line(
            points={{100,-60},{80,-60},{80,-10}},
            color={0,127,255}));

        connect(damRet.y, yRet)
          annotation (Line(points={{68,8.88178e-16},{-68,8.88178e-16},{-68,120}},
                                                              color={0,0,127}));
        connect(yOut, damOut.y)
          annotation (Line(points={{0,120},{0,72}}, color={0,0,127}));
        connect(yExh, damExh.y) annotation (Line(points={{60,120},{60,20},{-30,20},{-30,
                -48}}, color={0,0,127}));
        annotation (                       Icon(coordinateSystem(preserveAspectRatio=true,  extent={{-100,
                  -100},{100,100}}), graphics={
              Rectangle(
                extent={{-94,12},{90,0}},
                lineColor={0,0,255},
                fillColor={0,0,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{-94,-54},{96,-66}},
                lineColor={0,0,255},
                fillColor={0,0,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{-4,6},{6,-56}},
                lineColor={0,0,255},
                fillColor={0,0,255},
                fillPattern=FillPattern.Solid),
              Polygon(
                points={{-86,-12},{-64,24},{-46,24},{-70,-12},{-86,-12}},
                lineColor={0,0,0},
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid),
              Polygon(
                points={{48,12},{70,6},{48,0},{48,12}},
                lineColor={0,0,0},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{72,-58},{92,-62}},
                lineColor={0,0,0},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid),
              Polygon(
                points={{72,-54},{48,-60},{72,-66},{72,-54}},
                lineColor={0,0,0},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{28,8},{48,4}},
                lineColor={0,0,0},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid),
              Polygon(
                points={{-74,-76},{-52,-40},{-34,-40},{-58,-76},{-74,-76}},
                lineColor={0,0,0},
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid),
              Polygon(
                points={{-20,-40},{2,-4},{20,-4},{-4,-40},{-20,-40}},
                lineColor={0,0,0},
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{78,66},{90,10}},
                lineColor={0,0,255},
                fillColor={0,0,255},
                fillPattern=FillPattern.Solid),
              Rectangle(
                extent={{-94,66},{-82,8}},
                lineColor={0,0,255},
                fillColor={0,0,255},
                fillPattern=FillPattern.Solid),
              Line(
                points={{0,100},{0,60},{-54,60},{-54,24}},
                color={0,0,255}),  Text(
                extent={{-50,-84},{48,-132}},
                lineColor={0,0,255},
                textString=
                     "%name"),
              Line(
                points={{-68,100},{-68,80},{-20,80},{-20,-22}},
                color={0,0,255}),
              Line(
                points={{70,100},{70,-84},{-60,-84}},
                color={0,0,255})}),
      defaultComponentName="eco",
      Documentation(revisions="<html>
<ul>
<li>
November 10, 2017, by Michael Wetter:<br/>
Changed default of <code>raiseTime</code> as air damper motors, such as from JCI
have a travel time of about 30 seconds.
Shorter travel time also makes control loops more stable.
</li>
<li>
March 24, 2017, by Michael Wetter:<br/>
Renamed <code>filteredInput</code> to <code>use_inputFilter</code>.<br/>
This is for
<a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/665\">#665</a>.
</li>
<li>
March 22, 2017, by Michael Wetter:<br/>
Removed the assignments of <code>AOut</code>, <code>AExh</code> and <code>ARec</code> as these are done in the damper instance using
a final assignment of the parameter.
This allows scaling the model with <code>m_flow_nominal</code>,
which is generally known in the flow leg,
and <code>v_nominal</code>, for which a default value can be specified.<br/>
This is for
<a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/544\">#544</a>.
</li>
<li>
January 22, 2016, by Michael Wetter:<br/>
Corrected type declaration of pressure difference.
This is
for <a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/404\">#404</a>.
</li>
<li>
December 14, 2012 by Michael Wetter:<br/>
Renamed protected parameters for consistency with the naming conventions.
</li>
<li>
February 14, 2012 by Michael Wetter:<br/>
Added filter to approximate the travel time of the actuator.
</li>
<li>
February 3, 2012, by Michael Wetter:<br/>
Removed assignment of <code>m_flow_small</code> as it is no
longer used in its base class.
</li>
<li>
February 23, 2010 by Michael Wetter:<br/>
First implementation.
</li>
</ul>
</html>",       info="<html>
<p>
Model of an outside air mixing box with air dampers.
Set <code>y=0</code> to close the outside air and exhast air dampers.
</p>
<p>
If <code>dp_nominalIncludesDamper=true</code>, then the parameter <code>dp_nominal</code>
is equal to the pressure drop of the damper plus the fixed flow resistance at the nominal
flow rate.
If <code>dp_nominalIncludesDamper=false</code>, then <code>dp_nominal</code>
does not include the flow resistance of the air damper.
</p>
</html>"));
      end MixingBox;

      partial model PartialOpenLoop
        "Partial model of variable air volume flow system with terminal reheat and five thermal zones"

        package MediumA = Buildings.Media.Air "Medium model for air";
        package MediumW = Buildings.Media.Water "Medium model for water";

        constant Integer numZon=5 "Total number of served VAV boxes";
        parameter Modelica.SIunits.ThermalConductance UA_nominal(min=0)=
              -designCoolLoad/Buildings.Fluid.HeatExchangers.BaseClasses.lmtd(
              T_a1=26.2,
              T_b1=12.8,
              T_a2=6,
              T_b2=12)
          "Thermal conductance at nominal flow, used to compute heat capacity"
          annotation (Dialog(tab="General", group="Nominal condition"));
        parameter Modelica.SIunits.Volume VRooCor=AFloCor*flo.hRoo
          "Room volume corridor";
        parameter Modelica.SIunits.Volume VRooSou=AFloSou*flo.hRoo
          "Room volume south";
        parameter Modelica.SIunits.Volume VRooNor=AFloNor*flo.hRoo
          "Room volume north";
        parameter Modelica.SIunits.Volume VRooEas=AFloEas*flo.hRoo "Room volume east";
        parameter Modelica.SIunits.Volume VRooWes=AFloWes*flo.hRoo "Room volume west";

        parameter Modelica.SIunits.Area AFloCor=flo.cor.AFlo "Floor area corridor";
        parameter Modelica.SIunits.Area AFloSou=flo.sou.AFlo "Floor area south";
        parameter Modelica.SIunits.Area AFloNor=flo.nor.AFlo "Floor area north";
        parameter Modelica.SIunits.Area AFloEas=flo.eas.AFlo "Floor area east";
        parameter Modelica.SIunits.Area AFloWes=flo.wes.AFlo "Floor area west";

        parameter Modelica.SIunits.Area AFlo[numZon]={flo.cor.AFlo,flo.sou.AFlo,flo.eas.AFlo,
            flo.nor.AFlo,flo.wes.AFlo} "Floor area of each zone";
        final parameter Modelica.SIunits.Area ATot=sum(AFlo) "Total floor area";

        constant Real conv=1.2/3600 "Conversion factor for nominal mass flow rate";
        parameter Modelica.SIunits.MassFlowRate mCor_flow_nominal=6*VRooCor*conv
          "Design mass flow rate core";
        parameter Modelica.SIunits.MassFlowRate mSou_flow_nominal=6*VRooSou*conv
          "Design mass flow rate perimeter 1";
        parameter Modelica.SIunits.MassFlowRate mEas_flow_nominal=9*VRooEas*conv
          "Design mass flow rate perimeter 2";
        parameter Modelica.SIunits.MassFlowRate mNor_flow_nominal=6*VRooNor*conv
          "Design mass flow rate perimeter 3";
        parameter Modelica.SIunits.MassFlowRate mWes_flow_nominal=7*VRooWes*conv
          "Design mass flow rate perimeter 4";
        parameter Modelica.SIunits.MassFlowRate m_flow_nominal=0.7*(mCor_flow_nominal
             + mSou_flow_nominal + mEas_flow_nominal + mNor_flow_nominal +
            mWes_flow_nominal) "Nominal mass flow rate";
        parameter Modelica.SIunits.Angle lat=41.98*3.14159/180 "Latitude";

        parameter Modelica.SIunits.Temperature THeaOn=293.15
          "Heating setpoint during on";
        parameter Modelica.SIunits.Temperature THeaOff=285.15
          "Heating setpoint during off";
        parameter Modelica.SIunits.Temperature TCooOn=297.15
          "Cooling setpoint during on";
        parameter Modelica.SIunits.Temperature TCooOff=303.15
          "Cooling setpoint during off";
        parameter Modelica.SIunits.PressureDifference dpBuiStaSet(min=0) = 12
          "Building static pressure";
        parameter Real yFanMin = 0.1 "Minimum fan speed";

      //  parameter Modelica.SIunits.HeatFlowRate QHeaCoi_nominal= 2.5*yFanMin*m_flow_nominal*1000*(20 - 4)
      //    "Nominal capacity of heating coil";

        parameter Boolean allowFlowReversal=true
          "= false to simplify equations, assuming, but not enforcing, no flow reversal"
          annotation (Evaluate=true);

        parameter Boolean use_windPressure=true "Set to true to enable wind pressure";

        parameter Boolean sampleModel=true
          "Set to true to time-sample the model, which can give shorter simulation time if there is already time sampling in the system model"
          annotation (Evaluate=true, Dialog(tab=
                "Experimental (may be changed in future releases)"));

        // sizing parameter
        parameter Modelica.SIunits.HeatFlowRate designCoolLoad = -m_flow_nominal*1000*15 "Design cooling load";
        parameter Modelica.SIunits.HeatFlowRate designHeatLoad = 0.6*m_flow_nominal*1006*(18 - 8) "Design heating load";

        Buildings.Fluid.Sources.Outside amb(redeclare package Medium = MediumA,
            nPorts=3) "Ambient conditions"
          annotation (Placement(transformation(extent={{-136,-56},{-114,-34}})));
      //  Buildings.Fluid.HeatExchangers.DryCoilCounterFlow heaCoi(
      //    redeclare package Medium1 = MediumW,
      //    redeclare package Medium2 = MediumA,
      //    UA_nominal = QHeaCoi_nominal/Buildings.Fluid.HeatExchangers.BaseClasses.lmtd(
      //      T_a1=45,
      //      T_b1=35,
      //      T_a2=3,
      //      T_b2=20),
      //    m2_flow_nominal=m_flow_nominal,
      //    allowFlowReversal1=false,
      //    allowFlowReversal2=allowFlowReversal,
      //    dp1_nominal=0,
      //    dp2_nominal=200 + 200 + 100 + 40,
      //    m1_flow_nominal=QHeaCoi_nominal/4200/10,
      //    energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial)
      //    "Heating coil"
      //    annotation (Placement(transformation(extent={{118,-36},{98,-56}})));

        Buildings.Fluid.HeatExchangers.DryCoilEffectivenessNTU heaCoi(
          redeclare package Medium1 = MediumW,
          redeclare package Medium2 = MediumA,
          m1_flow_nominal=designHeatLoad/4200/5,
          m2_flow_nominal=0.6*m_flow_nominal,
          configuration=Buildings.Fluid.Types.HeatExchangerConfiguration.CounterFlow,
          Q_flow_nominal=designHeatLoad,
          dp1_nominal=30000,
          dp2_nominal=200 + 200 + 100 + 40,
          allowFlowReversal1=false,
          allowFlowReversal2=allowFlowReversal,
          T_a1_nominal=318.15,
          T_a2_nominal=281.65) "Heating coil"
          annotation (Placement(transformation(extent={{118,-36},{98,-56}})));

        Buildings.Fluid.HeatExchangers.WetCoilCounterFlow cooCoi(
          UA_nominal=UA_nominal,
          redeclare package Medium1 = MediumW,
          redeclare package Medium2 = MediumA,
          m1_flow_nominal=-designCoolLoad/4200/6,
          m2_flow_nominal=m_flow_nominal,
          dp2_nominal=0,
          dp1_nominal=30000,
          energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
          allowFlowReversal1=false,
          allowFlowReversal2=allowFlowReversal) "Cooling coil"
          annotation (Placement(transformation(extent={{210,-36},{190,-56}})));
        Buildings.Fluid.FixedResistances.PressureDrop dpRetDuc(
          m_flow_nominal=m_flow_nominal,
          redeclare package Medium = MediumA,
          allowFlowReversal=allowFlowReversal,
          dp_nominal=490)
                         "Pressure drop for return duct"
          annotation (Placement(transformation(extent={{400,130},{380,150}})));
        Buildings.Fluid.Movers.SpeedControlled_y fanSup(
          redeclare package Medium = MediumA,
          per(pressure(V_flow=m_flow_nominal/1.2*{0.2,0.6,1.0,1.2}, dp=(1030 + 220 +
                  10 + 20 + dpBuiStaSet)*{1.2,1.1,1.0,0.6})),
          energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
          addPowerToMedium=false) "Supply air fan"
          annotation (Placement(transformation(extent={{300,-50},{320,-30}})));

        Buildings.Fluid.Sensors.VolumeFlowRate senSupFlo(redeclare package
            Medium =
              MediumA, m_flow_nominal=m_flow_nominal)
          "Sensor for supply fan flow rate"
          annotation (Placement(transformation(extent={{400,-50},{420,-30}})));

        Buildings.Fluid.Sensors.VolumeFlowRate senRetFlo(redeclare package
            Medium =
              MediumA, m_flow_nominal=m_flow_nominal)
          "Sensor for return fan flow rate"
          annotation (Placement(transformation(extent={{360,130},{340,150}})));

        Buildings.Fluid.Sources.Boundary_pT sinHea(
          redeclare package Medium = MediumW,
          p=300000,
          T=313.15,
          nPorts=1) "Sink for heating coil" annotation (Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={80,-122})));
        Buildings.Fluid.Sources.Boundary_pT sinCoo(
          redeclare package Medium = MediumW,
          p=300000,
          T=285.15,
          nPorts=1) "Sink for cooling coil" annotation (Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={180,-120})));
        Modelica.Blocks.Routing.RealPassThrough TOut(y(
            final quantity="ThermodynamicTemperature",
            final unit="K",
            displayUnit="degC",
            min=0))
          annotation (Placement(transformation(extent={{-300,170},{-280,190}})));
        Buildings.Fluid.Sensors.TemperatureTwoPort TSup(
          redeclare package Medium = MediumA,
          m_flow_nominal=m_flow_nominal,
          allowFlowReversal=allowFlowReversal)
          annotation (Placement(transformation(extent={{330,-50},{350,-30}})));
        Buildings.Fluid.Sensors.RelativePressure dpDisSupFan(redeclare package
            Medium =
              MediumA) "Supply fan static discharge pressure" annotation (Placement(
              transformation(
              extent={{-10,10},{10,-10}},
              rotation=90,
              origin={320,0})));
        Buildings.Controls.SetPoints.OccupancySchedule occSch(occupancy=3600*{6,19})
          "Occupancy schedule"
          annotation (Placement(transformation(extent={{-318,-220},{-298,-200}})));
        Buildings.Utilities.Math.Min min(nin=5) "Computes lowest room temperature"
          annotation (Placement(transformation(extent={{1200,440},{1220,460}})));
        Buildings.Utilities.Math.Average ave(nin=5)
          "Compute average of room temperatures"
          annotation (Placement(transformation(extent={{1200,410},{1220,430}})));
        Buildings.Fluid.Sources.MassFlowSource_T souCoo(
          redeclare package Medium = MediumW,
          T=279.15,
          nPorts=1,
          use_m_flow_in=true) "Source for cooling coil" annotation (Placement(
              transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={230,-120})));
        Buildings.Fluid.Sensors.TemperatureTwoPort TRet(
          redeclare package Medium = MediumA,
          m_flow_nominal=m_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Return air temperature sensor"
          annotation (Placement(transformation(extent={{110,130},{90,150}})));
        Buildings.Fluid.Sensors.TemperatureTwoPort TMix(
          redeclare package Medium = MediumA,
          m_flow_nominal=m_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Mixed air temperature sensor"
          annotation (Placement(transformation(extent={{30,-50},{50,-30}})));
        Buildings.Fluid.Sources.MassFlowSource_T souHea(
          redeclare package Medium = MediumW,
          T=318.15,
          use_m_flow_in=true,
          nPorts=1)           "Source for heating coil" annotation (Placement(
              transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={132,-120})));
        Buildings.Fluid.Sensors.VolumeFlowRate VOut1(redeclare package Medium =
              MediumA, m_flow_nominal=m_flow_nominal) "Outside air volume flow rate"
          annotation (Placement(transformation(extent={{-72,-44},{-50,-22}})));

        FiveZoneAir.VAVReheat.ThermalZones.VAVBranch cor(
          redeclare package MediumA = MediumA,
          redeclare package MediumW = MediumW,
          m_flow_nominal=mCor_flow_nominal,
          VRoo=VRooCor,
          allowFlowReversal=allowFlowReversal)
          "Zone for core of buildings (azimuth will be neglected)"
          annotation (Placement(transformation(extent={{570,22},{610,62}})));
        FiveZoneAir.VAVReheat.ThermalZones.VAVBranch sou(
          redeclare package MediumA = MediumA,
          redeclare package MediumW = MediumW,
          m_flow_nominal=mSou_flow_nominal,
          VRoo=VRooSou,
          allowFlowReversal=allowFlowReversal) "South-facing thermal zone"
          annotation (Placement(transformation(extent={{750,20},{790,60}})));
        FiveZoneAir.VAVReheat.ThermalZones.VAVBranch eas(
          redeclare package MediumA = MediumA,
          redeclare package MediumW = MediumW,
          m_flow_nominal=mEas_flow_nominal,
          VRoo=VRooEas,
          allowFlowReversal=allowFlowReversal) "East-facing thermal zone"
          annotation (Placement(transformation(extent={{930,20},{970,60}})));
        FiveZoneAir.VAVReheat.ThermalZones.VAVBranch nor(
          redeclare package MediumA = MediumA,
          redeclare package MediumW = MediumW,
          m_flow_nominal=mNor_flow_nominal,
          VRoo=VRooNor,
          allowFlowReversal=allowFlowReversal) "North-facing thermal zone"
          annotation (Placement(transformation(extent={{1090,20},{1130,60}})));
        FiveZoneAir.VAVReheat.ThermalZones.VAVBranch wes(
          redeclare package MediumA = MediumA,
          redeclare package MediumW = MediumW,
          m_flow_nominal=mWes_flow_nominal,
          VRoo=VRooWes,
          allowFlowReversal=allowFlowReversal) "West-facing thermal zone"
          annotation (Placement(transformation(extent={{1290,20},{1330,60}})));
        Buildings.Fluid.FixedResistances.Junction splRetRoo1(
          redeclare package Medium = MediumA,
          m_flow_nominal={m_flow_nominal,m_flow_nominal - mCor_flow_nominal,
              mCor_flow_nominal},
          from_dp=false,
          linearized=true,
          energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
          dp_nominal(each displayUnit="Pa") = {0,0,0},
          portFlowDirection_1=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving,
          portFlowDirection_2=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering,
          portFlowDirection_3=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering)
          "Splitter for room return"
          annotation (Placement(transformation(extent={{630,10},{650,-10}})));
        Buildings.Fluid.FixedResistances.Junction splRetSou(
          redeclare package Medium = MediumA,
          m_flow_nominal={mSou_flow_nominal + mEas_flow_nominal + mNor_flow_nominal
               + mWes_flow_nominal,mEas_flow_nominal + mNor_flow_nominal +
              mWes_flow_nominal,mSou_flow_nominal},
          from_dp=false,
          linearized=true,
          energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
          dp_nominal(each displayUnit="Pa") = {0,0,0},
          portFlowDirection_1=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving,
          portFlowDirection_2=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering,
          portFlowDirection_3=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering)
          "Splitter for room return"
          annotation (Placement(transformation(extent={{812,10},{832,-10}})));
        Buildings.Fluid.FixedResistances.Junction splRetEas(
          redeclare package Medium = MediumA,
          m_flow_nominal={mEas_flow_nominal + mNor_flow_nominal + mWes_flow_nominal,
              mNor_flow_nominal + mWes_flow_nominal,mEas_flow_nominal},
          from_dp=false,
          linearized=true,
          energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
          dp_nominal(each displayUnit="Pa") = {0,0,0},
          portFlowDirection_1=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving,
          portFlowDirection_2=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering,
          portFlowDirection_3=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering)
          "Splitter for room return"
          annotation (Placement(transformation(extent={{992,10},{1012,-10}})));
        Buildings.Fluid.FixedResistances.Junction splRetNor(
          redeclare package Medium = MediumA,
          m_flow_nominal={mNor_flow_nominal + mWes_flow_nominal,mWes_flow_nominal,
              mNor_flow_nominal},
          from_dp=false,
          linearized=true,
          energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
          dp_nominal(each displayUnit="Pa") = {0,0,0},
          portFlowDirection_1=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving,
          portFlowDirection_2=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering,
          portFlowDirection_3=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering)
          "Splitter for room return"
          annotation (Placement(transformation(extent={{1142,10},{1162,-10}})));
        Buildings.Fluid.FixedResistances.Junction splSupRoo1(
          redeclare package Medium = MediumA,
          m_flow_nominal={m_flow_nominal,m_flow_nominal - mCor_flow_nominal,
              mCor_flow_nominal},
          from_dp=true,
          linearized=true,
          energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
          dp_nominal(each displayUnit="Pa") = {0,0,0},
          portFlowDirection_1=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering,
          portFlowDirection_2=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving,
          portFlowDirection_3=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving)
          "Splitter for room supply"
          annotation (Placement(transformation(extent={{570,-30},{590,-50}})));
        Buildings.Fluid.FixedResistances.Junction splSupSou(
          redeclare package Medium = MediumA,
          m_flow_nominal={mSou_flow_nominal + mEas_flow_nominal + mNor_flow_nominal
               + mWes_flow_nominal,mEas_flow_nominal + mNor_flow_nominal +
              mWes_flow_nominal,mSou_flow_nominal},
          from_dp=true,
          linearized=true,
          energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
          dp_nominal(each displayUnit="Pa") = {0,0,0},
          portFlowDirection_1=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering,
          portFlowDirection_2=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving,
          portFlowDirection_3=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving)
          "Splitter for room supply"
          annotation (Placement(transformation(extent={{750,-30},{770,-50}})));
        Buildings.Fluid.FixedResistances.Junction splSupEas(
          redeclare package Medium = MediumA,
          m_flow_nominal={mEas_flow_nominal + mNor_flow_nominal + mWes_flow_nominal,
              mNor_flow_nominal + mWes_flow_nominal,mEas_flow_nominal},
          from_dp=true,
          linearized=true,
          energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
          dp_nominal(each displayUnit="Pa") = {0,0,0},
          portFlowDirection_1=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering,
          portFlowDirection_2=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving,
          portFlowDirection_3=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving)
          "Splitter for room supply"
          annotation (Placement(transformation(extent={{930,-30},{950,-50}})));
        Buildings.Fluid.FixedResistances.Junction splSupNor(
          redeclare package Medium = MediumA,
          m_flow_nominal={mNor_flow_nominal + mWes_flow_nominal,mWes_flow_nominal,
              mNor_flow_nominal},
          from_dp=true,
          linearized=true,
          energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
          dp_nominal(each displayUnit="Pa") = {0,0,0},
          portFlowDirection_1=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Entering,
          portFlowDirection_2=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving,
          portFlowDirection_3=if allowFlowReversal then Modelica.Fluid.Types.PortFlowDirection.Bidirectional
               else Modelica.Fluid.Types.PortFlowDirection.Leaving)
          "Splitter for room supply"
          annotation (Placement(transformation(extent={{1090,-30},{1110,-50}})));
        Buildings.BoundaryConditions.WeatherData.ReaderTMY3 weaDat(filNam=
              Modelica.Utilities.Files.loadResource(
              "modelica://Buildings/Resources/weatherdata/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.mos"))
          annotation (Placement(transformation(extent={{-360,170},{-340,190}})));
        Buildings.BoundaryConditions.WeatherData.Bus weaBus "Weather Data Bus"
          annotation (Placement(transformation(extent={{-330,170},{-310,190}}),
              iconTransformation(extent={{-360,170},{-340,190}})));
        ThermalZones.Floor flo(
          redeclare final package Medium = MediumA,
          final lat=lat,
          final use_windPressure=use_windPressure,
          final sampleModel=sampleModel)
          "Model of a floor of the building that is served by this VAV system"
          annotation (Placement(transformation(extent={{772,396},{1100,616}})));
        Modelica.Blocks.Routing.DeMultiplex5 TRooAir(u(each unit="K", each
              displayUnit="degC")) "Demultiplex for room air temperature"
          annotation (Placement(transformation(extent={{490,160},{510,180}})));

        Buildings.Fluid.Sensors.TemperatureTwoPort TSupCor(
          redeclare package Medium = MediumA,
          initType=Modelica.Blocks.Types.Init.InitialState,
          m_flow_nominal=mCor_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Discharge air temperature"
          annotation (Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={580,92})));
        Buildings.Fluid.Sensors.TemperatureTwoPort TSupSou(
          redeclare package Medium = MediumA,
          initType=Modelica.Blocks.Types.Init.InitialState,
          m_flow_nominal=mSou_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Discharge air temperature"
          annotation (Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={760,92})));
        Buildings.Fluid.Sensors.TemperatureTwoPort TSupEas(
          redeclare package Medium = MediumA,
          initType=Modelica.Blocks.Types.Init.InitialState,
          m_flow_nominal=mEas_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Discharge air temperature"
          annotation (Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={940,90})));
        Buildings.Fluid.Sensors.TemperatureTwoPort TSupNor(
          redeclare package Medium = MediumA,
          initType=Modelica.Blocks.Types.Init.InitialState,
          m_flow_nominal=mNor_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Discharge air temperature"
          annotation (Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={1100,94})));
        Buildings.Fluid.Sensors.TemperatureTwoPort TSupWes(
          redeclare package Medium = MediumA,
          initType=Modelica.Blocks.Types.Init.InitialState,
          m_flow_nominal=mWes_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Discharge air temperature"
          annotation (Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={1300,90})));
        Buildings.Fluid.Sensors.VolumeFlowRate VSupCor_flow(
          redeclare package Medium = MediumA,
          initType=Modelica.Blocks.Types.Init.InitialState,
          m_flow_nominal=mCor_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Discharge air flow rate" annotation (
            Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={580,130})));
        Buildings.Fluid.Sensors.VolumeFlowRate VSupSou_flow(
          redeclare package Medium = MediumA,
          initType=Modelica.Blocks.Types.Init.InitialState,
          m_flow_nominal=mSou_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Discharge air flow rate" annotation (
            Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={760,130})));
        Buildings.Fluid.Sensors.VolumeFlowRate VSupEas_flow(
          redeclare package Medium = MediumA,
          initType=Modelica.Blocks.Types.Init.InitialState,
          m_flow_nominal=mEas_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Discharge air flow rate" annotation (
            Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={940,128})));
        Buildings.Fluid.Sensors.VolumeFlowRate VSupNor_flow(
          redeclare package Medium = MediumA,
          initType=Modelica.Blocks.Types.Init.InitialState,
          m_flow_nominal=mNor_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Discharge air flow rate" annotation (
            Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={1100,132})));
        Buildings.Fluid.Sensors.VolumeFlowRate VSupWes_flow(
          redeclare package Medium = MediumA,
          initType=Modelica.Blocks.Types.Init.InitialState,
          m_flow_nominal=mWes_flow_nominal,
          allowFlowReversal=allowFlowReversal) "Discharge air flow rate" annotation (
            Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=90,
              origin={1300,128})));
        FiveZoneAir.VAVReheat.BaseClasses.MixingBox eco(
          redeclare package Medium = MediumA,
          mOut_flow_nominal=m_flow_nominal,
          dpOut_nominal=10,
          mRec_flow_nominal=m_flow_nominal,
          dpRec_nominal=10,
          mExh_flow_nominal=m_flow_nominal,
          dpExh_nominal=10,
          from_dp=false) "Economizer" annotation (Placement(transformation(
              extent={{-10,-10},{10,10}},
              rotation=0,
              origin={-10,-46})));

        Results res(
          final A=ATot,
          PFan=fanSup.P + 0,
          PHea=heaCoi.Q2_flow + cor.terHea.Q1_flow + nor.terHea.Q1_flow + wes.terHea.Q1_flow
               + eas.terHea.Q1_flow + sou.terHea.Q1_flow,
          PCooSen=cooCoi.QSen2_flow,
          PCooLat=cooCoi.QLat2_flow) "Results of the simulation";
        /*fanRet*/

      protected
        model Results "Model to store the results of the simulation"
          parameter Modelica.SIunits.Area A "Floor area";
          input Modelica.SIunits.Power PFan "Fan energy";
          input Modelica.SIunits.Power PHea "Heating energy";
          input Modelica.SIunits.Power PCooSen "Sensible cooling energy";
          input Modelica.SIunits.Power PCooLat "Latent cooling energy";

          Real EFan(
            unit="J/m2",
            start=0,
            nominal=1E5,
            fixed=true) "Fan energy";
          Real EHea(
            unit="J/m2",
            start=0,
            nominal=1E5,
            fixed=true) "Heating energy";
          Real ECooSen(
            unit="J/m2",
            start=0,
            nominal=1E5,
            fixed=true) "Sensible cooling energy";
          Real ECooLat(
            unit="J/m2",
            start=0,
            nominal=1E5,
            fixed=true) "Latent cooling energy";
          Real ECoo(unit="J/m2") "Total cooling energy";
        equation

          A*der(EFan) = PFan;
          A*der(EHea) = PHea;
          A*der(ECooSen) = PCooSen;
          A*der(ECooLat) = PCooLat;
          ECoo = ECooSen + ECooLat;

        end Results;
      public
        Buildings.Controls.OBC.CDL.Continuous.Gain gaiHeaCoi(k=designHeatLoad/4200/5)
                        "Gain for heating coil mass flow rate"
          annotation (Placement(transformation(extent={{100,-220},{120,-200}})));
        Buildings.Controls.OBC.CDL.Continuous.Gain gaiCooCoi(k=-designCoolLoad/4200/6)
                        "Gain for cooling coil mass flow rate"
          annotation (Placement(transformation(extent={{100,-258},{120,-238}})));
        Buildings.Controls.OBC.CDL.Logical.OnOffController freSta(bandwidth=1)
          "Freeze stat for heating coil"
          annotation (Placement(transformation(extent={{0,-102},{20,-82}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant freStaTSetPoi(k=273.15
               + 3) "Freeze stat set point for heating coil"
          annotation (Placement(transformation(extent={{-40,-96},{-20,-76}})));
      equation
        connect(fanSup.port_b, dpDisSupFan.port_a) annotation (Line(
            points={{320,-40},{320,-10}},
            color={0,0,0},
            smooth=Smooth.None,
            pattern=LinePattern.Dot));
        connect(TSup.port_a, fanSup.port_b) annotation (Line(
            points={{330,-40},{320,-40}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(amb.ports[1], VOut1.port_a) annotation (Line(
            points={{-114,-42.0667},{-94,-42.0667},{-94,-33},{-72,-33}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splRetRoo1.port_1, dpRetDuc.port_a) annotation (Line(
            points={{630,0},{430,0},{430,140},{400,140}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splRetNor.port_1, splRetEas.port_2) annotation (Line(
            points={{1142,0},{1110,0},{1110,0},{1078,0},{1078,0},{1012,0}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splRetEas.port_1, splRetSou.port_2) annotation (Line(
            points={{992,0},{952,0},{952,0},{912,0},{912,0},{832,0}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splRetSou.port_1, splRetRoo1.port_2) annotation (Line(
            points={{812,0},{650,0}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splSupRoo1.port_3, cor.port_a) annotation (Line(
            points={{580,-30},{580,22}},
            color={0,127,255},
            thickness=0.5));
        connect(splSupRoo1.port_2, splSupSou.port_1) annotation (Line(
            points={{590,-40},{750,-40}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splSupSou.port_3, sou.port_a) annotation (Line(
            points={{760,-30},{760,20}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splSupSou.port_2, splSupEas.port_1) annotation (Line(
            points={{770,-40},{930,-40}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splSupEas.port_3, eas.port_a) annotation (Line(
            points={{940,-30},{940,20}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splSupEas.port_2, splSupNor.port_1) annotation (Line(
            points={{950,-40},{1090,-40}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splSupNor.port_3, nor.port_a) annotation (Line(
            points={{1100,-30},{1100,20}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(splSupNor.port_2, wes.port_a) annotation (Line(
            points={{1110,-40},{1300,-40},{1300,20}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(cooCoi.port_b1, sinCoo.ports[1]) annotation (Line(
            points={{190,-52},{180,-52},{180,-110}},
            color={28,108,200},
            thickness=0.5));
        connect(weaDat.weaBus, weaBus) annotation (Line(
            points={{-340,180},{-320,180}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(weaBus.TDryBul, TOut.u) annotation (Line(
            points={{-320,180},{-302,180}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(amb.weaBus, weaBus) annotation (Line(
            points={{-136,-44.78},{-320,-44.78},{-320,180}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(splRetRoo1.port_3, flo.portsCor[2]) annotation (Line(
            points={{640,10},{640,364},{874,364},{874,472},{898,472},{898,
                449.533},{924.286,449.533}},
            color={0,127,255},
            thickness=0.5));
        connect(splRetSou.port_3, flo.portsSou[2]) annotation (Line(
            points={{822,10},{822,350},{900,350},{900,420.2},{924.286,420.2}},
            color={0,127,255},
            thickness=0.5));
        connect(splRetEas.port_3, flo.portsEas[2]) annotation (Line(
            points={{1002,10},{1002,368},{1067.2,368},{1067.2,445.867}},
            color={0,127,255},
            thickness=0.5));
        connect(splRetNor.port_3, flo.portsNor[2]) annotation (Line(
            points={{1152,10},{1152,446},{924.286,446},{924.286,478.867}},
            color={0,127,255},
            thickness=0.5));
        connect(splRetNor.port_2, flo.portsWes[2]) annotation (Line(
            points={{1162,0},{1342,0},{1342,394},{854,394},{854,449.533}},
            color={0,127,255},
            thickness=0.5));
        connect(weaBus, flo.weaBus) annotation (Line(
            points={{-320,180},{-320,506},{988.714,506}},
            color={255,204,51},
            thickness=0.5,
            smooth=Smooth.None));
        connect(flo.TRooAir, min.u) annotation (Line(
            points={{1094.14,491.333},{1164.7,491.333},{1164.7,450},{1198,450}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(flo.TRooAir, ave.u) annotation (Line(
            points={{1094.14,491.333},{1166,491.333},{1166,420},{1198,420}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));
        connect(TRooAir.u, flo.TRooAir) annotation (Line(
            points={{488,170},{480,170},{480,538},{1164,538},{1164,491.333},{
                1094.14,491.333}},
            color={0,0,127},
            smooth=Smooth.None,
            pattern=LinePattern.Dash));

        connect(cooCoi.port_b2, fanSup.port_a) annotation (Line(
            points={{210,-40},{300,-40}},
            color={0,127,255},
            smooth=Smooth.None,
            thickness=0.5));
        connect(cor.port_b, TSupCor.port_a) annotation (Line(
            points={{580,62},{580,82}},
            color={0,127,255},
            thickness=0.5));

        connect(sou.port_b, TSupSou.port_a) annotation (Line(
            points={{760,60},{760,82}},
            color={0,127,255},
            thickness=0.5));
        connect(eas.port_b, TSupEas.port_a) annotation (Line(
            points={{940,60},{940,80}},
            color={0,127,255},
            thickness=0.5));
        connect(nor.port_b, TSupNor.port_a) annotation (Line(
            points={{1100,60},{1100,84}},
            color={0,127,255},
            thickness=0.5));
        connect(wes.port_b, TSupWes.port_a) annotation (Line(
            points={{1300,60},{1300,80}},
            color={0,127,255},
            thickness=0.5));

        connect(TSupCor.port_b, VSupCor_flow.port_a) annotation (Line(
            points={{580,102},{580,120}},
            color={0,127,255},
            thickness=0.5));
        connect(TSupSou.port_b, VSupSou_flow.port_a) annotation (Line(
            points={{760,102},{760,120}},
            color={0,127,255},
            thickness=0.5));
        connect(TSupEas.port_b, VSupEas_flow.port_a) annotation (Line(
            points={{940,100},{940,100},{940,118}},
            color={0,127,255},
            thickness=0.5));
        connect(TSupNor.port_b, VSupNor_flow.port_a) annotation (Line(
            points={{1100,104},{1100,122}},
            color={0,127,255},
            thickness=0.5));
        connect(TSupWes.port_b, VSupWes_flow.port_a) annotation (Line(
            points={{1300,100},{1300,118}},
            color={0,127,255},
            thickness=0.5));
        connect(VSupCor_flow.port_b, flo.portsCor[1]) annotation (Line(
            points={{580,140},{580,372},{866,372},{866,480},{912.571,480},{
                912.571,449.533}},
            color={0,127,255},
            thickness=0.5));

        connect(VSupSou_flow.port_b, flo.portsSou[1]) annotation (Line(
            points={{760,140},{760,356},{912.571,356},{912.571,420.2}},
            color={0,127,255},
            thickness=0.5));
        connect(VSupEas_flow.port_b, flo.portsEas[1]) annotation (Line(
            points={{940,138},{940,376},{1055.49,376},{1055.49,445.867}},
            color={0,127,255},
            thickness=0.5));
        connect(VSupNor_flow.port_b, flo.portsNor[1]) annotation (Line(
            points={{1100,142},{1100,498},{912.571,498},{912.571,478.867}},
            color={0,127,255},
            thickness=0.5));
        connect(VSupWes_flow.port_b, flo.portsWes[1]) annotation (Line(
            points={{1300,138},{1300,384},{842.286,384},{842.286,449.533}},
            color={0,127,255},
            thickness=0.5));
        connect(VOut1.port_b, eco.port_Out) annotation (Line(
            points={{-50,-33},{-42,-33},{-42,-40},{-20,-40}},
            color={0,127,255},
            thickness=0.5));
        connect(eco.port_Sup, TMix.port_a) annotation (Line(
            points={{0,-40},{30,-40}},
            color={0,127,255},
            thickness=0.5));
        connect(eco.port_Exh, amb.ports[2]) annotation (Line(
            points={{-20,-52},{-96,-52},{-96,-45},{-114,-45}},
            color={0,127,255},
            thickness=0.5));
        connect(eco.port_Ret, TRet.port_b) annotation (Line(
            points={{0,-52},{10,-52},{10,140},{90,140}},
            color={0,127,255},
            thickness=0.5));
        connect(senRetFlo.port_a, dpRetDuc.port_b)
          annotation (Line(points={{360,140},{380,140}}, color={0,127,255}));
        connect(TSup.port_b, senSupFlo.port_a)
          annotation (Line(points={{350,-40},{400,-40}}, color={0,127,255}));
        connect(senSupFlo.port_b, splSupRoo1.port_1)
          annotation (Line(points={{420,-40},{570,-40}}, color={0,127,255}));
        connect(cooCoi.port_a1, souCoo.ports[1]) annotation (Line(
            points={{210,-52},{230,-52},{230,-110}},
            color={28,108,200},
            thickness=0.5));
        connect(gaiHeaCoi.y, souHea.m_flow_in) annotation (Line(points={{122,-210},{
                124,-210},{124,-132}}, color={0,0,127}));
        connect(gaiCooCoi.y, souCoo.m_flow_in) annotation (Line(points={{122,-248},{
                222,-248},{222,-132}}, color={0,0,127}));
        connect(dpDisSupFan.port_b, amb.ports[3]) annotation (Line(
            points={{320,10},{320,14},{-88,14},{-88,-47.9333},{-114,-47.9333}},
            color={0,0,0},
            pattern=LinePattern.Dot));
        connect(senRetFlo.port_b, TRet.port_a) annotation (Line(points={{340,140},{
                226,140},{110,140}}, color={0,127,255}));
        connect(freStaTSetPoi.y, freSta.reference)
          annotation (Line(points={{-18,-86},{-2,-86}}, color={0,0,127}));
        connect(freSta.u, TMix.T) annotation (Line(points={{-2,-98},{-10,-98},{-10,-70},
                {20,-70},{20,-20},{40,-20},{40,-29}}, color={0,0,127}));
        connect(TMix.port_b, heaCoi.port_a2) annotation (Line(
            points={{50,-40},{98,-40}},
            color={0,127,255},
            thickness=0.5));
        connect(heaCoi.port_b2, cooCoi.port_a2) annotation (Line(
            points={{118,-40},{190,-40}},
            color={0,127,255},
            thickness=0.5));
        connect(souHea.ports[1], heaCoi.port_a1) annotation (Line(
            points={{132,-110},{132,-52},{118,-52}},
            color={28,108,200},
            thickness=0.5));
        connect(heaCoi.port_b1, sinHea.ports[1]) annotation (Line(
            points={{98,-52},{80,-52},{80,-112}},
            color={28,108,200},
            thickness=0.5));
        annotation (Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-380,
                  -400},{1420,600}})), Documentation(info="<html>
<p>
This model consist of an HVAC system, a building envelope model and a model
for air flow through building leakage and through open doors.
</p>
<p>
The HVAC system is a variable air volume (VAV) flow system with economizer
and a heating and cooling coil in the air handler unit. There is also a
reheat coil and an air damper in each of the five zone inlet branches.
The figure below shows the schematic diagram of the HVAC system
</p>
<p align=\"center\">
<img alt=\"image\" src=\"modelica://Buildings/Resources/Images/Examples/VAVReheat/vavSchematics.png\" border=\"1\"/>
</p>
<p>
Most of the HVAC control in this model is open loop.
Two models that extend this model, namely
<a href=\"modelica://Buildings.Examples.VAVReheat.ASHRAE2006\">
Buildings.Examples.VAVReheat.ASHRAE2006</a>
and
<a href=\"modelica://Buildings.Examples.VAVReheat.Guideline36\">
Buildings.Examples.VAVReheat.Guideline36</a>
add closed loop control. See these models for a description of
the control sequence.
</p>
<p>
To model the heat transfer through the building envelope,
a model of five interconnected rooms is used.
The five room model is representative of one floor of the
new construction medium office building for Chicago, IL,
as described in the set of DOE Commercial Building Benchmarks
(Deru et al, 2009). There are four perimeter zones and one core zone.
The envelope thermal properties meet ASHRAE Standard 90.1-2004.
The thermal room model computes transient heat conduction through
walls, floors and ceilings and long-wave radiative heat exchange between
surfaces. The convective heat transfer coefficient is computed based
on the temperature difference between the surface and the room air.
There is also a layer-by-layer short-wave radiation,
long-wave radiation, convection and conduction heat transfer model for the
windows. The model is similar to the
Window 5 model and described in TARCOG 2006.
</p>
<p>
Each thermal zone can have air flow from the HVAC system, through leakages of the building envelope (except for the core zone) and through bi-directional air exchange through open doors that connect adjacent zones. The bi-directional air exchange is modeled based on the differences in static pressure between adjacent rooms at a reference height plus the difference in static pressure across the door height as a function of the difference in air density.
Infiltration is a function of the
flow imbalance of the HVAC system.
</p>
<h4>References</h4>
<p>
Deru M., K. Field, D. Studer, K. Benne, B. Griffith, P. Torcellini,
 M. Halverson, D. Winiarski, B. Liu, M. Rosenberg, J. Huang, M. Yazdanian, and D. Crawley.
<i>DOE commercial building research benchmarks for commercial buildings</i>.
Technical report, U.S. Department of Energy, Energy Efficiency and
Renewable Energy, Office of Building Technologies, Washington, DC, 2009.
</p>
<p>
TARCOG 2006: Carli, Inc., TARCOG: Mathematical models for calculation
of thermal performance of glazing systems with our without
shading devices, Technical Report, Oct. 17, 2006.
</p>
</html>",       revisions="<html>
<ul>
<li>
September 26, 2017, by Michael Wetter:<br/>
Separated physical model from control to facilitate implementation of alternate control
sequences.
</li>
<li>
May 19, 2016, by Michael Wetter:<br/>
Changed chilled water supply temperature to <i>6&deg;C</i>.
This is
for <a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/509\">#509</a>.
</li>
<li>
April 26, 2016, by Michael Wetter:<br/>
Changed controller for freeze protection as the old implementation closed
the outdoor air damper during summer.
This is
for <a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/511\">#511</a>.
</li>
<li>
January 22, 2016, by Michael Wetter:<br/>
Corrected type declaration of pressure difference.
This is
for <a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/404\">#404</a>.
</li>
<li>
September 24, 2015 by Michael Wetter:<br/>
Set default temperature for medium to avoid conflicting
start values for alias variables of the temperature
of the building and the ambient air.
This is for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/426\">issue 426</a>.
</li>
</ul>
</html>"));
      end PartialOpenLoop;

      partial model EnergyMeterAirSide "System example for airside system"

       Modelica.Blocks.Sources.RealExpression eleSupFan             "Pow of fan"
          annotation (Placement(transformation(extent={{1220,580},{1240,600}})));
       parameter Real cooCOP = 3.5 "Coefficient of performance for 
    the water plant system (assume constant for differt loads)";

        Modelica.Blocks.Sources.RealExpression elePla
          "Power from the waterside HVAC component"
          annotation (Placement(transformation(extent={{1220,560},{1240,580}})));
        Modelica.Blocks.Sources.RealExpression eleCoiVAV
          "Power of VAV terminal reheat coil"
          annotation (Placement(transformation(extent={{1220,602},{1240,622}})));
        Modelica.Blocks.Sources.RealExpression gasBoi
          "Gas consumption of gas boiler"
          annotation (Placement(transformation(extent={{1220,534},{1240,554}})));
        Modelica.Blocks.Math.MultiSum eleTot(nu=3) "Electricity in total"
          annotation (Placement(transformation(extent={{1284,606},{1296,618}})));

        Modelica.Blocks.Continuous.Integrator eleTotInt
          annotation (Placement(transformation(extent={{1320,602},{1340,622}})));
        Modelica.Blocks.Continuous.Integrator gasTotInt
          annotation (Placement(transformation(extent={{1320,534},{1340,554}})));
      equation
        connect(eleCoiVAV.y, eleTot.u[1]) annotation (Line(points={{1241,612},{1262,612},
                {1262,614.8},{1284,614.8}},      color={0,0,127}));
        connect(eleSupFan.y, eleTot.u[2]) annotation (Line(points={{1241,590},{1262.5,
                590},{1262.5,612},{1284,612}},     color={0,0,127}));
        connect(elePla.y, eleTot.u[3]) annotation (Line(points={{1241,570},{1264,570},
                {1264,609.2},{1284,609.2}}, color={0,0,127}));
        connect(eleTot.y, eleTotInt.u)
          annotation (Line(points={{1297.02,612},{1318,612}}, color={0,0,127}));
        connect(gasBoi.y, gasTotInt.u)
          annotation (Line(points={{1241,544},{1318,544}}, color={0,0,127}));
        annotation (Diagram(coordinateSystem(extent={{-100,-100},{1580,700}})), Icon(
              coordinateSystem(extent={{-100,-100},{1580,700}})));
      end EnergyMeterAirSide;

      partial model ZoneAirTemperatureDeviation
        "Calculate the zone air temperature deviation outside the boundary"

         FiveZoneAir.VAVReheat.BaseClasses.BandDeviationSum banDevSum[5](each
            uppThreshold=25 + 273.15, each lowThreshold=23 + 273.15)
          annotation (Placement(transformation(extent={{1240,480},{1260,500}})));
        Modelica.Blocks.Math.MultiSum TAirDev(nu=5) "Zone air temperature deviation in total"
          annotation (Placement(transformation(extent={{1282,484},{1294,496}})));
        Modelica.Blocks.Continuous.Integrator TAirTotDev
          annotation (Placement(transformation(extent={{1318,480},{1338,500}})));
      equation
        connect(TAirDev.y,TAirTotDev. u)
          annotation (Line(points={{1295.02,490},{1316,490}}, color={0,0,127}));
        connect(banDevSum.TDev,TAirDev. u[1:5]) annotation (Line(points={{1261,490},
                {1272,490},{1272,486.64},{1282,486.64}}, color={0,0,127}));
        annotation (Diagram(coordinateSystem(extent={{-100,-100},{1580,700}})), Icon(
              coordinateSystem(extent={{-100,-100},{1580,700}})));
      end ZoneAirTemperatureDeviation;

      model BandDeviationSum "Model to calculate the signal that out of the band"

        parameter Real uppThreshold(unit="K") "Comparison with respect to upper threshold";
        parameter Real lowThreshold(unit="K") "Comparison with respect to lower threshold";
        Modelica.Blocks.Logical.GreaterThreshold greThr(threshold=uppThreshold)
          annotation (Placement(transformation(extent={{-72,40},{-52,60}})));
        Modelica.Blocks.Interfaces.RealInput u1
                                      "Connector of Boolean input signal"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Logical.LessThreshold lesThr(threshold=lowThreshold)
          annotation (Placement(transformation(extent={{-72,-60},{-52,-40}})));

        Modelica.Blocks.Logical.Switch swi1
          annotation (Placement(transformation(extent={{-10,10},{10,30}})));
        Modelica.Blocks.Sources.Constant const(k=0)
          annotation (Placement(transformation(extent={{-72,-10},{-52,10}})));
        Modelica.Blocks.Sources.RealExpression reaExpUpp(y=abs(u1 - uppThreshold))
          annotation (Placement(transformation(extent={{-72,14},{-52,34}})));
        Modelica.Blocks.Logical.Switch swi2
          annotation (Placement(transformation(extent={{-12,-40},{8,-20}})));
        Modelica.Blocks.Sources.RealExpression reaExpLow(y=abs(u1 - lowThreshold))
          annotation (Placement(transformation(extent={{-72,-36},{-52,-16}})));
        Modelica.Blocks.Math.Add add
          annotation (Placement(transformation(extent={{26,-10},{46,10}})));
        Modelica.Blocks.Logical.Switch swi3
          annotation (Placement(transformation(extent={{-10,60},{10,80}})));
        Modelica.Blocks.Interfaces.RealOutput TDev
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));
        Buildings.Controls.OBC.CDL.Interfaces.BooleanInput uSupFan
          "Supply fan status"
          annotation (Placement(transformation(extent={{-140,60},{-100,100}}),
          iconTransformation(extent={{-140,-80},{-100,-40}})));
      equation
        connect(greThr.u, u1) annotation (Line(points={{-74,50},{-92,50},{-92,0},{-120,
                0}}, color={0,0,127}));
        connect(greThr.y, swi1.u2) annotation (Line(points={{-51,50},{-32,50},{-32,20},
                {-12,20}}, color={255,0,255}));
        connect(const.y, swi1.u3) annotation (Line(points={{-51,0},{-32,0},{-32,12},{-12,
                12}}, color={0,0,127}));
        connect(reaExpUpp.y, swi1.u1) annotation (Line(points={{-51,24},{-32,24},{-32,
                28},{-12,28}}, color={0,0,127}));
        connect(lesThr.y, swi2.u2) annotation (Line(points={{-51,-50},{-36,-50},{-36,-30},
                {-14,-30}}, color={255,0,255}));
        connect(const.y, swi2.u3) annotation (Line(points={{-51,0},{-32,0},{-32,-38},{
                -14,-38}}, color={0,0,127}));
        connect(reaExpLow.y, swi2.u1) annotation (Line(points={{-51,-26},{-32,-26},{-32,
                -22},{-14,-22}}, color={0,0,127}));
        connect(u1, lesThr.u) annotation (Line(points={{-120,0},{-92,0},{-92,-50},{-74,
                -50}}, color={0,0,127}));
        connect(swi1.y, add.u1)
          annotation (Line(points={{11,20},{16,20},{16,6},{24,6}}, color={0,0,127}));
        connect(swi2.y, add.u2) annotation (Line(points={{9,-30},{16,-30},{16,-6},{24,
                -6}}, color={0,0,127}));
        connect(swi3.y, TDev) annotation (Line(points={{11,70},{80,70},{80,0},{110,0}},
              color={0,0,127}));
        connect(const.y, swi3.u3) annotation (Line(points={{-51,0},{-20,0},{-20,62},{-12,
                62}}, color={0,0,127}));
        connect(swi3.u2, uSupFan) annotation (Line(points={{-12,70},{-40,70},{-40,80},
                {-120,80}}, color={255,0,255}));
        connect(add.y, swi3.u1) annotation (Line(points={{47,0},{60,0},{60,40},{-26,40},
                {-26,78},{-12,78}}, color={0,0,127}));
        annotation (Icon(coordinateSystem(preserveAspectRatio=false), graphics={
                Rectangle(
                lineColor={0,0,0},
                extent={{-100,100},{100,-100}},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid), Text(
                extent={{-100,148},{100,102}},
                lineColor={0,0,255},
                textString="%name")}),                                 Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          __Dymola_Commands);
      end BandDeviationSum;

      model BandDeviationSumTest
        extends Modelica.Icons.Example;

        FiveZoneAir.VAVReheat.BaseClasses.BandDeviationSum bandDevSum(
            uppThreshold=26 + 273.15, lowThreshold=22 + 273.15)
          annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
        Modelica.Blocks.Sources.Sine sine(
          amplitude=5,
          freqHz=0.0005,
          offset=23.5 + 273.15,
          phase=0,
          startTime=0)
          annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
        Modelica.Blocks.Sources.BooleanPulse booPul(period=1000, startTime=400)
          annotation (Placement(transformation(extent={{-60,30},{-40,50}})));
      equation
        connect(sine.y, bandDevSum.u1)
          annotation (Line(points={{-39,0},{-12,0}}, color={0,0,127}));
        connect(booPul.y, bandDevSum.uSupFan) annotation (Line(points={{-39,40},{-28,
                40},{-28,-6},{-12,-6}}, color={255,0,255}));
        annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          experiment(StopTime=3600, __Dymola_Algorithm="Cvode"));
      end BandDeviationSumTest;
    annotation (preferredView="info", Documentation(info="<html>
<p>
This package contains base classes that are used to construct the models in
<a href=\"modelica://Buildings.Examples.VAVReheat\">Buildings.Examples.VAVReheat</a>.
</p>
</html>"));
    end BaseClasses;
  end VAVReheat;
annotation (uses(
    Buildings(version="7.0.0"),
    Modelica(version="3.2.3"),
      Complex(version="3.2.3"),
      Modelica_LinearSystems2(version="2.3.5"),
      Modelica_Synchronous(version="0.93.0")),
  version="1.0.0",
  conversion(noneFromVersion=""));
end FiveZoneAir;
