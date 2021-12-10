within ;
package FiveZone "Five zone VAV supervisory control"
  model Guideline36TSup
    "Variable air volume flow system with terminal reheat and five thermal zones"
    extends Modelica.Icons.Example;
    extends FiveZone.VAVReheat.BaseClasses.PartialOpenLoop(flo(
        cor(T_start=273.15 + 24),
        eas(T_start=273.15 + 24),
        sou(T_start=273.15 + 24),
        wes(T_start=273.15 + 24),
        nor(T_start=273.15 + 24)));
    extends FiveZone.VAVReheat.BaseClasses.EnergyMeterAirSide(
      eleCoiVAV(y=cor.terHea.Q1_flow + nor.terHea.Q1_flow + wes.terHea.Q1_flow +
            eas.terHea.Q1_flow + sou.terHea.Q1_flow),
      eleSupFan(y=fanSup.P),
      elePla(y=cooCoi.Q1_flow/cooCOP),
      gasBoi(y=-heaCoi.Q1_flow));
    extends FiveZone.VAVReheat.BaseClasses.ZoneAirTemperatureDeviation(
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
    FiveZone.VAVReheat.Controls.Controller conAHU(
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
    Modelica.Blocks.Interfaces.RealOutput PHVAC(
       quantity="Power",
       unit="W")
      "Power consumption of HVAC equipment, W"
      annotation (Placement(transformation(extent={{1400,650},{1420,670}})));
    Modelica.Blocks.Interfaces.RealOutput PBoiGas(
       quantity="Power",
       unit="W")
      "Boiler gas consumption, W"
      annotation (Placement(transformation(extent={{1400,576},{1420,596}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirSou( unit="K",
        displayUnit="degC")
      "Room air temperatures, K; 1- South, 2- East, 3- North, 4- West, 5- Core;"
      annotation (Placement(transformation(extent={{1400,438},{1420,458}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirDevTot
      "Total zone air temperature deviation, K*s"
      annotation (Placement(transformation(extent={{1400,480},{1420,500}})));
    Modelica.Blocks.Interfaces.RealOutput EHVACTot(
       quantity="Energy",
       unit="J")
      "Total electricity energy consumption of HVAC equipment, J"
      annotation (Placement(transformation(extent={{1400,602},{1420,622}})));
    Modelica.Blocks.Interfaces.RealOutput EGasTot(
       quantity="Energy",
       unit="J")
      "Total boiler gas consumption, J"
      annotation (Placement(transformation(extent={{1400,534},{1420,554}})));
    Modelica.Blocks.Interfaces.RealOutput TAirOut(
    unit="K",  displayUnit=
         "degC") "Outdoor air temperature"
      annotation (Placement(transformation(extent={{1400,-260},{1420,-240}})));
    Modelica.Blocks.Interfaces.RealOutput GHI(
       quantity="RadiantEnergyFluenceRate",
       unit="W/m2")
      "Global horizontal solar radiation, W/m2"
      annotation (Placement(transformation(extent={{1400,-302},{1420,-282}})));
    Buildings.Controls.OBC.CDL.Interfaces.RealInput uTSupSet(
      final unit="K",
      final displayUnit="degC",
      final quantity="ThermodynamicTemperature")
      "External supply air temperature setpoint"
      annotation (Placement(transformation(extent={{-422,236},{-382,276}})));
    Buildings.Utilities.Math.Min minyDam(nin=5)
      "Computes lowest zone damper position"
      annotation (Placement(transformation(extent={{1352,-102},{1372,-82}})));
    Modelica.Blocks.Interfaces.RealOutput yDamMin "Minimum VAV damper position"
      annotation (Placement(transformation(extent={{1400,-102},{1420,-82}})));
    Modelica.Blocks.Interfaces.RealOutput yDamMax "Minimum VAV damper position"
      annotation (Placement(transformation(extent={{1400,-168},{1420,-148}})));
    Buildings.Utilities.Math.Max maxyDam(nin=5)
      annotation (Placement(transformation(extent={{1356,-168},{1376,-148}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirEas( unit="K",
        displayUnit="degC")
      "Room air temperatures, K; 1- South, 2- East, 3- North, 4- West, 5- Core;"
      annotation (Placement(transformation(extent={{1400,406},{1420,426}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirNor( unit="K",
        displayUnit="degC")
      "Room air temperatures, K; 1- South, 2- East, 3- North, 4- West, 5- Core;"
      annotation (Placement(transformation(extent={{1400,376},{1420,396}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirWes( unit="K",
        displayUnit="degC")
      "Room air temperatures, K; 1- South, 2- East, 3- North, 4- West, 5- Core;"
      annotation (Placement(transformation(extent={{1400,346},{1420,366}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirCor( unit="K",
        displayUnit="degC")
      "Room air temperatures, K; 1- South, 2- East, 3- North, 4- West, 5- Core;"
      annotation (Placement(transformation(extent={{1400,318},{1420,338}})));
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
    connect(eleTot.y, PHVAC) annotation (Line(points={{1297.02,612},{1306,612},{1306,
            660},{1410,660}}, color={0,0,127}));
    connect(gasBoi.y, gasTotInt.u)
      annotation (Line(points={{1241,544},{1318,544}}, color={0,0,127}));
    connect(gasBoi.y, PBoiGas) annotation (Line(points={{1241,544},{1306,544},{1306,
            586},{1410,586}}, color={0,0,127}));
    connect(TAirTotDev.y, TRooAirDevTot)
      annotation (Line(points={{1339,490},{1410,490}}, color={0,0,127}));
    connect(eleTotInt.y, EHVACTot)
      annotation (Line(points={{1341,612},{1410,612}}, color={0,0,127}));
    connect(gasTotInt.y, EGasTot)
      annotation (Line(points={{1341,544},{1410,544}}, color={0,0,127}));
    connect(weaBus.TDryBul, TAirOut) annotation (Line(
        points={{-320,180},{-314,180},{-314,-282},{1342,-282},{1342,-250},{1410,-250}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-3,6},{-3,6}},
        horizontalAlignment=TextAlignment.Right));

    connect(weaBus.HGloHor, GHI) annotation (Line(
        points={{-320,180},{-318,180},{-318,-292},{1410,-292}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(conAHU.uTSupSet, uTSupSet) annotation (Line(points={{336,635.6},{-364,
            635.6},{-364,256},{-402,256}}, color={0,0,127}));
    connect(cor.y_actual, minyDam.u[1]) annotation (Line(
        points={{612,58},{618,58},{618,-90},{1350,-90},{1350,-93.6}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(sou.y_actual, minyDam.u[2]) annotation (Line(
        points={{792,56},{798,56},{798,-88},{1350,-88},{1350,-92.8}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(eas.y_actual, minyDam.u[3]) annotation (Line(
        points={{972,56},{976,56},{976,-92},{1350,-92}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(nor.y_actual, minyDam.u[4]) annotation (Line(
        points={{1132,56},{1136,56},{1136,-92},{1350,-92},{1350,-91.2}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(wes.y_actual, minyDam.u[5]) annotation (Line(
        points={{1332,56},{1334,56},{1334,-88},{1350,-88},{1350,-90.4}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(minyDam.y, yDamMin)
      annotation (Line(points={{1373,-92},{1410,-92}}, color={0,0,127}));
    connect(maxyDam.y, yDamMax)
      annotation (Line(points={{1377,-158},{1410,-158}}, color={0,0,127}));
    connect(maxyDam.u[1], cor.y_actual) annotation (Line(
        points={{1354,-159.6},{618,-159.6},{618,58},{612,58}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(sou.y_actual, maxyDam.u[2]) annotation (Line(
        points={{792,56},{800,56},{800,-158},{818,-158},{818,-158.8},{1354,-158.8}},
        color={0,0,127},
        pattern=LinePattern.Dash));

    connect(eas.y_actual, maxyDam.u[3]) annotation (Line(
        points={{972,56},{982,56},{982,-158},{1354,-158}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(nor.y_actual, maxyDam.u[4]) annotation (Line(
        points={{1132,56},{1132,-156},{1354,-156},{1354,-157.2}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(wes.y_actual, maxyDam.u[5]) annotation (Line(
        points={{1332,56},{1336,56},{1336,-156},{1354,-156},{1354,-156.4}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(flo.TRooAir[1], TRooAirSou) annotation (Line(points={{1094.14,488.4},
            {1124,488.4},{1124,468},{1322,468},{1322,448},{1410,448}}, color={0,0,
            127}));
    connect(flo.TRooAir[2], TRooAirEas) annotation (Line(points={{1094.14,
            489.867},{1130,489.867},{1130,472},{1326,472},{1326,416},{1410,416}},
                                                                         color={0,
            0,127}));
    connect(flo.TRooAir[3], TRooAirNor) annotation (Line(points={{1094.14,
            491.333},{1136,491.333},{1136,470},{1322,470},{1322,386},{1410,386}},
                                                                         color={0,
            0,127}));
    connect(flo.TRooAir[4], TRooAirWes) annotation (Line(points={{1094.14,492.8},
            {1130,492.8},{1130,470},{1318,470},{1318,356},{1410,356}}, color={0,0,
            127}));
    connect(flo.TRooAir[5], TRooAirCor) annotation (Line(points={{1094.14,
            494.267},{1128,494.267},{1128,472},{1334,472},{1334,328},{1410,328}},
                                                                         color={0,
            0,127}));
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
        StartTime=19180800,
        StopTime=19785600,
        Tolerance=1e-06,
        __Dymola_Algorithm="Cvode"),
      Icon(coordinateSystem(extent={{-100,-100},{100,100}})));
  end Guideline36TSup;

  model Guideline36Baseline
    "Variable air volume flow system with terminal reheat and five thermal zones"
    extends Modelica.Icons.Example;
    extends FiveZone.VAVReheat.BaseClasses.PartialOpenLoop(flo(
        cor(T_start=273.15 + 24),
        eas(T_start=273.15 + 24),
        sou(T_start=273.15 + 24),
        wes(T_start=273.15 + 24),
        nor(T_start=273.15 + 24)));
    extends FiveZone.VAVReheat.BaseClasses.EnergyMeterAirSide(
      eleCoiVAV(y=cor.terHea.Q1_flow + nor.terHea.Q1_flow + wes.terHea.Q1_flow +
            eas.terHea.Q1_flow + sou.terHea.Q1_flow),
      eleSupFan(y=fanSup.P),
      elePla(y=cooCoi.Q1_flow/cooCOP),
      gasBoi(y=-heaCoi.Q1_flow));
    extends FiveZone.VAVReheat.BaseClasses.ZoneAirTemperatureDeviation(
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
    Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.Controller conAHU(
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
    Modelica.Blocks.Interfaces.RealOutput PHVAC(
       quantity="Power",
       unit="W")
      "Power consumption of HVAC equipment, W"
      annotation (Placement(transformation(extent={{1400,650},{1420,670}})));
    Modelica.Blocks.Interfaces.RealOutput PBoiGas(
       quantity="Power",
       unit="W")
      "Boiler gas consumption, W"
      annotation (Placement(transformation(extent={{1400,576},{1420,596}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirSou(
        unit="K",
        displayUnit="degC")
      "Room air temperatures, K; 1- South, 2- East, 3- North, 4- West, 5- Core;"
      annotation (Placement(transformation(extent={{1400,438},{1420,458}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirDevTot
      "Total zone air temperature deviation, K*s"
      annotation (Placement(transformation(extent={{1400,480},{1420,500}})));
    Modelica.Blocks.Interfaces.RealOutput EHVACTot(
       quantity="Energy",
       unit="J")
      "Total electricity energy consumption of HVAC equipment, J"
      annotation (Placement(transformation(extent={{1400,602},{1420,622}})));
    Modelica.Blocks.Interfaces.RealOutput EGasTot(
       quantity="Energy",
       unit="J")
      "Total boiler gas consumption, J"
      annotation (Placement(transformation(extent={{1400,534},{1420,554}})));
    Modelica.Blocks.Interfaces.RealOutput TAirOut(
    unit="K",  displayUnit=
         "degC") "Outdoor air temperature"
      annotation (Placement(transformation(extent={{1400,-260},{1420,-240}})));
    Modelica.Blocks.Interfaces.RealOutput GHI(
       quantity="RadiantEnergyFluenceRate",
       unit="W/m2")
      "Global horizontal solar radiation, W/m2"
      annotation (Placement(transformation(extent={{1400,-302},{1420,-282}})));
    Buildings.Utilities.Math.Min minyDam(nin=5)
      "Computes lowest zone damper position"
      annotation (Placement(transformation(extent={{1352,-102},{1372,-82}})));
    Modelica.Blocks.Interfaces.RealOutput yDamMin "Minimum VAV damper position"
      annotation (Placement(transformation(extent={{1400,-102},{1420,-82}})));
    Modelica.Blocks.Interfaces.RealOutput yDamMax "Minimum VAV damper position"
      annotation (Placement(transformation(extent={{1400,-168},{1420,-148}})));
    Buildings.Utilities.Math.Max maxyDam(nin=5)
      annotation (Placement(transformation(extent={{1356,-168},{1376,-148}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirEas(
        unit="K",
        displayUnit="degC")
      "Room air temperatures, K; 1- South, 2- East, 3- North, 4- West, 5- Core;"
      annotation (Placement(transformation(extent={{1400,406},{1420,426}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirNor(
        unit="K",
        displayUnit="degC")
      "Room air temperatures, K; 1- South, 2- East, 3- North, 4- West, 5- Core;"
      annotation (Placement(transformation(extent={{1400,376},{1420,396}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirWes(
        unit="K",
        displayUnit="degC")
      "Room air temperatures, K; 1- South, 2- East, 3- North, 4- West, 5- Core;"
      annotation (Placement(transformation(extent={{1400,346},{1420,366}})));
    Modelica.Blocks.Interfaces.RealOutput TRooAirCor(
        unit="K",
        displayUnit="degC")
      "Room air temperatures, K; 1- South, 2- East, 3- North, 4- West, 5- Core;"
      annotation (Placement(transformation(extent={{1400,318},{1420,338}})));
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
    connect(eleTot.y, PHVAC) annotation (Line(points={{1297.02,612},{1306,612},{1306,
            660},{1410,660}}, color={0,0,127}));
    connect(gasBoi.y, gasTotInt.u)
      annotation (Line(points={{1241,544},{1318,544}}, color={0,0,127}));
    connect(gasBoi.y, PBoiGas) annotation (Line(points={{1241,544},{1306,544},{1306,
            586},{1410,586}}, color={0,0,127}));
    connect(TAirTotDev.y, TRooAirDevTot)
      annotation (Line(points={{1339,490},{1410,490}}, color={0,0,127}));
    connect(eleTotInt.y, EHVACTot)
      annotation (Line(points={{1341,612},{1410,612}}, color={0,0,127}));
    connect(gasTotInt.y, EGasTot)
      annotation (Line(points={{1341,544},{1410,544}}, color={0,0,127}));
    connect(weaBus.TDryBul, TAirOut) annotation (Line(
        points={{-320,180},{-314,180},{-314,-282},{1342,-282},{1342,-250},{1410,-250}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-3,6},{-3,6}},
        horizontalAlignment=TextAlignment.Right));

    connect(weaBus.HGloHor, GHI) annotation (Line(
        points={{-320,180},{-318,180},{-318,-292},{1410,-292}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(cor.y_actual, minyDam.u[1]) annotation (Line(
        points={{612,58},{618,58},{618,-90},{1350,-90},{1350,-93.6}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(sou.y_actual, minyDam.u[2]) annotation (Line(
        points={{792,56},{798,56},{798,-88},{1350,-88},{1350,-92.8}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(eas.y_actual, minyDam.u[3]) annotation (Line(
        points={{972,56},{976,56},{976,-92},{1350,-92}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(nor.y_actual, minyDam.u[4]) annotation (Line(
        points={{1132,56},{1136,56},{1136,-92},{1350,-92},{1350,-91.2}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(wes.y_actual, minyDam.u[5]) annotation (Line(
        points={{1332,56},{1334,56},{1334,-88},{1350,-88},{1350,-90.4}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(minyDam.y, yDamMin)
      annotation (Line(points={{1373,-92},{1410,-92}}, color={0,0,127}));
    connect(maxyDam.y, yDamMax)
      annotation (Line(points={{1377,-158},{1410,-158}}, color={0,0,127}));
    connect(maxyDam.u[1], cor.y_actual) annotation (Line(
        points={{1354,-159.6},{618,-159.6},{618,58},{612,58}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(sou.y_actual, maxyDam.u[2]) annotation (Line(
        points={{792,56},{800,56},{800,-158},{818,-158},{818,-158.8},{1354,-158.8}},
        color={0,0,127},
        pattern=LinePattern.Dash));

    connect(eas.y_actual, maxyDam.u[3]) annotation (Line(
        points={{972,56},{982,56},{982,-158},{1354,-158}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(nor.y_actual, maxyDam.u[4]) annotation (Line(
        points={{1132,56},{1132,-156},{1354,-156},{1354,-157.2}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(wes.y_actual, maxyDam.u[5]) annotation (Line(
        points={{1332,56},{1336,56},{1336,-156},{1354,-156},{1354,-156.4}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(flo.TRooAir[1], TRooAirSou) annotation (Line(points={{1094.14,488.4},
            {1124,488.4},{1124,468},{1322,468},{1322,448},{1410,448}}, color={0,0,
            127}));
    connect(flo.TRooAir[2], TRooAirEas) annotation (Line(points={{1094.14,
            489.867},{1130,489.867},{1130,472},{1326,472},{1326,416},{1410,416}},
                                                                         color={0,
            0,127}));
    connect(flo.TRooAir[3], TRooAirNor) annotation (Line(points={{1094.14,
            491.333},{1136,491.333},{1136,470},{1322,470},{1322,386},{1410,386}},
                                                                         color={0,
            0,127}));
    connect(flo.TRooAir[4], TRooAirWes) annotation (Line(points={{1094.14,492.8},
            {1130,492.8},{1130,470},{1318,470},{1318,356},{1410,356}}, color={0,0,
            127}));
    connect(flo.TRooAir[5], TRooAirCor) annotation (Line(points={{1094.14,
            494.267},{1128,494.267},{1128,472},{1334,472},{1334,328},{1410,328}},
                                                                         color={0,
            0,127}));
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
        StartTime=19180800,
        StopTime=19785600,
        Tolerance=1e-06,
        __Dymola_Algorithm="Cvode"),
      Icon(coordinateSystem(extent={{-100,-100},{100,100}})));
  end Guideline36Baseline;

  model SystemBaseline "System example for fault injection"
    extends Modelica.Icons.Example;
    extends FiveZone.BaseClasses.PartialHotWaterside(
      final Q_flow_boi_nominal=designHeatLoad,
      minFloBypHW(k=0.1),
      pumSpeHW(reset=Buildings.Types.Reset.Parameter, y_reset=0),
      boiTSup(
        y_start=0,
        reset=Buildings.Types.Reset.Parameter,
        y_reset=0),
      boi(show_T=false),
      triResHW(TMin=313.15, TMax=321.15));
    extends FiveZone.BaseClasses.PartialAirside(
      fanSup(show_T=false),
      conAHU(
        pNumIgnReq=1,
        TSupSetMin=284.95,
        numIgnReqSupTem=1,
        kTSup=0.5,
        TiTSup=120),
      conVAVWes(
        VDisSetMin_flow=0.05*conVAVWes.V_flow_nominal,
        VDisConMin_flow=0.05*conVAVWes.V_flow_nominal,
        errTZonCoo_1=0.8,
        errTZonCoo_2=0.4),
      conVAVCor(
        VDisSetMin_flow=0.05*conVAVCor.V_flow_nominal,
        VDisConMin_flow=0.05*conVAVCor.V_flow_nominal,
        errTZonCoo_1=0.8,
        errTZonCoo_2=0.4),
      conVAVSou(
        VDisSetMin_flow=0.05*conVAVSou.V_flow_nominal,
        VDisConMin_flow=0.05*conVAVSou.V_flow_nominal,
        errTZonCoo_1=0.8,
        errTZonCoo_2=0.4),
      conVAVEas(
        VDisSetMin_flow=0.05*conVAVEas.V_flow_nominal,
        VDisConMin_flow=0.05*conVAVEas.V_flow_nominal,
        errTZonCoo_1=0.8,
        errTZonCoo_2=0.4),
      conVAVNor(
        VDisSetMin_flow=0.05*conVAVNor.V_flow_nominal,
        VDisConMin_flow=0.05*conVAVNor.V_flow_nominal,
        errTZonCoo_1=0.8,
        errTZonCoo_2=0.4));
    extends FiveZone.BaseClasses.PartialWaterside(
      redeclare
        Buildings.Applications.DataCenters.ChillerCooled.Equipment.IntegratedPrimaryLoadSide
        chiWSE(
        use_inputFilter=true,
        addPowerToMedium=false,
        perPum=perPumPri),
      watVal(
        redeclare package Medium = MediumW,
        m_flow_nominal=m1_flow_chi_nominal,
        dpValve_nominal=6000,
        riseTime=60),
      final QEva_nominal=designCoolLoad,
      pumCW(use_inputFilter=true),
      resCHW(dp_nominal=139700),
      temDifPreRes(
        samplePeriod(displayUnit="s"),
        uTri=0.9,
        dpMin=0.5*dpSetPoi,
        dpMax=dpSetPoi,
        TMin(displayUnit="degC") = 278.15,
        TMax(displayUnit="degC") = 283.15),
      pumSpe(yMin=0.2));

    extends FiveZone.VAVReheat.BaseClasses.ZoneAirTemperatureDeviation(
        banDevSum(each uppThreshold=24.5 + 273.15, each lowThreshold=23.5 + 273.15));

    extends FiveZone.BaseClasses.EnergyMeter(
      eleCoiVAV(y=cor.terHea.Q1_flow + nor.terHea.Q1_flow + wes.terHea.Q1_flow +
            eas.terHea.Q1_flow + sou.terHea.Q1_flow),
      eleSupFan(y=fanSup.P),
      eleChi(y=chiWSE.powChi[1]),
      eleCHWP(y=chiWSE.powPum[1]),
      eleCWP(y=pumCW.P),
      eleHWP(y=pumHW.P),
      eleCT(y=cooTow.PFan),
      gasBoi(y=boi.QFue_flow));

    parameter Buildings.Fluid.Movers.Data.Generic[numChi] perPumPri(
      each pressure=Buildings.Fluid.Movers.BaseClasses.Characteristics.flowParameters(
            V_flow=m2_flow_chi_nominal/1000*{0.2,0.6,1.0,1.2},
            dp=(dp2_chi_nominal+dp2_wse_nominal+139700+36000)*{1.5,1.3,1.0,0.6}))
      "Performance data for primary pumps";

    FiveZone.Controls.CoolingMode cooModCon(
      tWai=1200,
      deaBan1=1.1,
      deaBan2=0.5,
      deaBan3=1.1,
      deaBan4=0.5) "Cooling mode controller"
      annotation (Placement(transformation(extent={{1028,-266},{1048,-246}})));
    Modelica.Blocks.Sources.RealExpression towTApp(y=cooTow.TWatOut_nominal -
          cooTow.TAirInWB_nominal)
      "Cooling tower approach temperature"
      annotation (Placement(transformation(extent={{988,-300},{1008,-280}})));
    Modelica.Blocks.Sources.RealExpression yVal5(y=if cooModCon.y == Integer(FiveZone.Types.CoolingModes.FullMechanical)
           then 1 else 0)
      "On/off signal for valve 5"
      annotation (Placement(transformation(extent={{1060,-192},{1040,-172}})));
    Modelica.Blocks.Sources.RealExpression yVal6(y=if cooModCon.y == Integer(FiveZone.Types.CoolingModes.FreeCooling)
           then 1 else 0)
      "On/off signal for valve 6"
      annotation (Placement(transformation(extent={{1060,-208},{1040,-188}})));
    Buildings.Controls.OBC.CDL.Continuous.Product proCHWP
      annotation (Placement(transformation(extent={{1376,-260},{1396,-240}})));

    FiveZone.Controls.PlantRequest plaReqChi
      annotation (Placement(transformation(extent={{1044,-120},{1064,-100}})));
    FiveZone.Controls.ChillerPlantEnableDisable chiPlaEnaDis(yFanSpeMin=0.15,
        plaReqTim=30*60)
      annotation (Placement(transformation(extent={{1100,-120},{1120,-100}})));
    Modelica.Blocks.Math.BooleanToReal booToRea
      annotation (Placement(transformation(extent={{1168,-126},{1188,-106}})));
    Buildings.Controls.OBC.CDL.Routing.BooleanReplicator booRepSupFan(final nout=
          numZon) "Replicate boolean input"
      annotation (Placement(transformation(extent={{500,634},{520,654}})));
    FiveZone.Controls.BoilerPlantEnableDisable boiPlaEnaDis(
      yFanSpeMin=0.15,
      plaReqTim=30*60,
      TOutPla=291.15)
      annotation (Placement(transformation(extent={{-278,-170},{-258,-150}})));
    Modelica.Blocks.Math.BooleanToReal booToReaHW
      annotation (Placement(transformation(extent={{-218,-170},{-198,-150}})));
    FiveZone.Controls.PlantRequest plaReqBoi
      annotation (Placement(transformation(extent={{-320,-170},{-300,-150}})));
    Buildings.Controls.OBC.CDL.Continuous.Product proHWVal
      annotation (Placement(transformation(extent={{40,-190},{60,-170}})));
    Buildings.Controls.OBC.CDL.Continuous.Product proCHWVal
      annotation (Placement(transformation(extent={{468,-118},{488,-98}})));

    Buildings.Utilities.IO.SignalExchange.Overwrite oveActTCHWSup(description="chilled water supply temperature setpoint", u(
        unit="K",
        min=273.15 + 5,
        max=273.15 + 10))
                     "Overwrite the chilled water supply temperature setpoint"
      annotation (Placement(transformation(extent={{1132,-280},{1152,-260}})));
    Buildings.Utilities.IO.SignalExchange.Read PHVAC(description="total HVAC power",
        KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
      y(unit="W"))
      "Read the total HVAC power"
      annotation (Placement(transformation(extent={{1320,694},{1340,714}})));
    Buildings.Utilities.IO.SignalExchange.Read PGas(description="total gas power",
        KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.GasPower,
      y(unit="W")) "Read the total gas power"
      annotation (Placement(transformation(extent={{1320,530},{1340,550}})));
    Buildings.Utilities.IO.SignalExchange.Read dtTZonAir(
      description=" total zone air temperature deviation",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.AirZoneTemperature,
      y(unit="K")) "Read the total zone air temperature deviation"
      annotation (Placement(transformation(extent={{1320,438},{1340,458}})));

    Buildings.Utilities.IO.SignalExchange.Read TOutAir(
      description=" outdoor air temperature",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="K")) "Read the outdoor air temperature"
      annotation (Placement(transformation(extent={{-300,120},{-280,140}})));

    Buildings.Utilities.IO.SignalExchange.Read GHI(
      description=" global horizontal solar radiation",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="W/m2")) "Read the global horizontal solar radiation"
      annotation (Placement(transformation(extent={{-300,80},{-280,100}})));

    Buildings.Utilities.Math.Min minyDam(nin=5)
      "Computes lowest zone damper position"
      annotation (Placement(transformation(extent={{1220,280},{1240,300}})));
    Buildings.Utilities.Math.Max maxyDam(nin=5)
      annotation (Placement(transformation(extent={{1220,242},{1240,262}})));
    Buildings.Utilities.IO.SignalExchange.Read yDamMin(
      description=" minimum zone air damper position",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="1")) "Read the minimum zone air damper position"
      annotation (Placement(transformation(extent={{1260,280},{1280,300}})));

    Buildings.Utilities.IO.SignalExchange.Read yDamMax(
      description=" maximum zone air damper position",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="1")) "Read the maximum zone air damper position"
      annotation (Placement(transformation(extent={{1260,242},{1280,262}})));

    Buildings.Utilities.IO.SignalExchange.Read yWatVal(
      description=" chilled water valve position",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="1")) "Read the chilled water valve position"
      annotation (Placement(transformation(extent={{468,-164},{488,-144}})));

    Buildings.Utilities.IO.SignalExchange.Read yCooTowFan(
      description="cooling tower fan speed",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="1")) "Read the cooling tower fan speed"
      annotation (Placement(transformation(extent={{740,-380},{760,-360}})));

    Buildings.Utilities.IO.SignalExchange.Read yFanSpe(
      description=" fan speed",
      KPIs=Buildings.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      y(unit="1")) "Read the fan speed"
      annotation (Placement(transformation(extent={{300,-120},{320,-100}})));

  equation

    connect(chiWSE.TCHWSupWSE,cooModCon. TCHWSupWSE)
      annotation (Line(
        points={{673,-212},{666,-212},{666,-76},{1016,-76},{1016,-260.444},{
            1026,-260.444}},
        color={0,0,127}));
    connect(towTApp.y,cooModCon. TApp)
      annotation (Line(
        points={{1009,-290},{1018,-290},{1018,-257.111},{1026,-257.111}},
        color={0,0,127}));
    connect(cooModCon.TCHWRetWSE, TCHWRet.T)
      annotation (Line(
        points={{1026,-263.778},{1014,-263.778},{1014,-66},{608,-66},{608,-177}},
      color={0,0,127}));
    connect(cooModCon.y, chiStaCon.cooMod)
      annotation (Line(
        points={{1049,-254.889},{1072,-254.889},{1072,-66},{1270,-66},{1270,
            -122},{1284,-122}},
        color={255,127,0}));
    connect(cooModCon.y,intToBoo.u)
      annotation (Line(
        points={{1049,-254.889},{1072,-254.889},{1072,-66},{1270,-66},{1270,
            -154},{1284,-154}},
        color={255,127,0}));
    connect(cooModCon.y, cooTowSpeCon.cooMod) annotation (Line(points={{1049,
            -254.889},{1072,-254.889},{1072,-66},{1270,-66},{1270,-93.5556},{
            1284,-93.5556}},                           color={255,127,0}));
    connect(cooModCon.y, CWPumCon.cooMod) annotation (Line(points={{1049,
            -254.889},{1072,-254.889},{1072,-66},{1270,-66},{1270,-201},{1282,
            -201}},                         color={255,127,0}));
    connect(yVal5.y, chiWSE.yVal5) annotation (Line(points={{1039,-182},{864,-182},
            {864,-211},{695.6,-211}},
                                color={0,0,127}));
    connect(watVal.port_a, cooCoi.port_b1) annotation (Line(points={{538,-98},{538,
            -86},{182,-86},{182,-52},{190,-52}},
                             color={0,127,255},
        thickness=0.5));
    connect(cooCoi.port_a1, TCHWSup.port_b) annotation (Line(points={{210,-52},{220,
            -52},{220,-78},{642,-78},{642,-128},{758,-128}},
                                         color={0,127,255},
        thickness=0.5));
    connect(proCHWP.y, chiWSE.yPum[1]) annotation (Line(points={{1398,-250},{1404,
            -250},{1404,-340},{704,-340},{704,-203.6},{695.6,-203.6}},
                                          color={0,0,127}));
    connect(weaBus.TWetBul, cooModCon.TWetBul) annotation (Line(
        points={{-320,180},{-320,22},{436,22},{436,-60},{1008,-60},{1008,
            -253.778},{1026,-253.778}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(weaBus.TWetBul, cooTow.TAir) annotation (Line(
        points={{-320,180},{-320,24},{434,24},{434,-60},{724,-60},{724,-312},{736,
            -312}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(TCWSup.T, cooTowSpeCon.TCWSup) annotation (Line(points={{828,-305},
            {828,-64},{1274,-64},{1274,-100.667},{1284,-100.667}},
          color={0,0,127}));
    connect(TCHWSup.T, cooTowSpeCon.TCHWSup) annotation (Line(points={{768,-117},
            {768,-64},{1272,-64},{1272,-104.222},{1284,-104.222}},
                              color={0,0,127}));
    connect(pumSpe.y, proCHWP.u2) annotation (Line(points={{1361,-248},{1366,-248},
            {1366,-256},{1374,-256}},
                                   color={0,0,127}));
    connect(watVal.y_actual, temDifPreRes.u) annotation (Line(points={{531,-113},{
            530,-113},{530,-122},{518,-122},{518,-72},{964,-72},{964,-242},{1088,-242}},
                                                            color={0,0,127}));
    connect(cooModCon.y, temDifPreRes.uOpeMod) annotation (Line(points={{1049,
            -254.889},{1072,-254.889},{1072,-236},{1088,-236}},
          color={255,127,0}));
    connect(TOut.y, chiPlaEnaDis.TOut) annotation (Line(points={{-279,180},{1078,
            180},{1078,-105.4},{1098,-105.4}},
                                           color={0,0,127}));
    connect(chiPlaEnaDis.ySupFan, conAHU.ySupFan) annotation (Line(points={{1098,
            -110},{1076,-110},{1076,629.333},{424,629.333}},             color={
            255,0,255}));
    connect(cooModCon.yPla, chiPlaEnaDis.yPla) annotation (Line(points={{1026,
            -247.333},{1022,-247.333},{1022,-70},{1142,-70},{1142,-110},{1121,
            -110}}, color={255,0,255}));
    connect(gai.y, pumCW.y) annotation (Line(points={{1347,-206},{1400,-206},{1400,
            -342},{880,-342},{880,-288},{898,-288}}, color={0,0,127}));
    connect(cooTowSpeCon.y, cooTow.y) annotation (Line(points={{1307,-97.1111},
            {1402,-97.1111},{1402,-344},{722,-344},{722,-308},{736,-308}},
                                                                        color={0,
            0,127}));
    connect(chiOn.y, chiWSE.on[1]) annotation (Line(points={{1347,-128},{1408,-128},
            {1408,-338},{868,-338},{868,-215.6},{695.6,-215.6}},
                                                              color={255,0,255}));
    connect(chiPlaEnaDis.yPla, booToRea.u)
      annotation (Line(points={{1121,-110},{1142,-110},{1142,-116},{1166,-116}},
                                                         color={255,0,255}));
    connect(booToRea.y, proCHWP.u1) annotation (Line(points={{1189,-116},{1246,-116},
            {1246,-64},{1368,-64},{1368,-244},{1374,-244}},
                    color={0,0,127}));
    connect(booToRea.y, val.y) annotation (Line(points={{1189,-116},{1246,-116},{1246,
            -174},{1420,-174},{1420,-342},{620,-342},{620,-296},{646,-296},{646,-304}},
                                                                    color={0,0,
            127}));
    connect(conAHU.ySupFan, andFreSta.u2) annotation (Line(points={{424,629.333},
            {436,629.333},{436,658},{-50,658},{-50,-138},{-22,-138}},
                        color={255,0,255}));
    connect(heaCoi.port_b1, HWVal.port_a)
      annotation (Line(points={{98,-52},{98,-170},{98,-170}},color={238,46,47},
        thickness=0.5));
    connect(boiPlaEnaDis.yPla, booToReaHW.u)
      annotation (Line(points={{-257,-160},{-220,-160}}, color={255,0,255}));
    connect(booToReaHW.y, boiIsoVal.y) annotation (Line(points={{-197,-160},{-182,
            -160},{-182,-360},{242,-360},{242,-300},{292,-300},{292,-308}},
                               color={0,0,127}));
    connect(booToReaHW.y, proPumHW.u1) annotation (Line(points={{-197,-160},{-178,
            -160},{-178,-72},{-98,-72},{-98,-210},{-42,-210},{-42,-308},{-34,-308}},
                                          color={0,0,127}));
    connect(booToReaHW.y, proBoi.u1) annotation (Line(points={{-197,-160},{-184,-160},
            {-184,-82},{-96,-82},{-96,-208},{-40,-208},{-40,-266},{-34,-266}},
                                          color={0,0,127}));
    connect(boiPlaEnaDis.yPla, pumSpeHW.trigger) annotation (Line(points={{-257,-160},
            {-240,-160},{-240,-82},{-92,-82},{-92,-338},{-68,-338},{-68,-332}},
                                                                  color={255,0,
            255}));
    connect(boiPlaEnaDis.yPla, minFloBypHW.yPla) annotation (Line(points={{-257,-160},
            {-240,-160},{-240,-80},{-92,-80},{-92,-251},{-76,-251}}, color={255,0,
            255}));
    connect(cooModCon.yPla, pumSpe.trigger) annotation (Line(points={{1026,
            -247.333},{1022,-247.333},{1022,-336},{1342,-336},{1342,-260}}, color=
           {255,0,255}));
    connect(THWSup.port_a, heaCoi.port_a1) annotation (Line(points={{350,-214},{350,
            -140},{142,-140},{142,-52},{118,-52}},     color={238,46,47},
        thickness=0.5));
    connect(wseOn.y, chiWSE.on[2]) annotation (Line(points={{1347,-154},{1408,-154},
            {1408,-338},{866,-338},{866,-215.6},{695.6,-215.6}},
                                                              color={255,0,255}));
    connect(boiPlaEnaDis.yPla, boiTSup.trigger) annotation (Line(points={{-257,-160},
            {-238,-160},{-238,-78},{-92,-78},{-92,-292},{-72,-292},{-72,-290}},
                                                                  color={255,0,
            255}));
    connect(plaReqChi.yPlaReq, chiPlaEnaDis.yPlaReq) annotation (Line(points={{1065,
            -110},{1072,-110},{1072,-114},{1098,-114}},      color={255,127,0}));
    connect(swiFreSta.y, plaReqBoi.uPlaVal) annotation (Line(points={{42,-130},{58,
            -130},{58,-70},{-250,-70},{-250,-120},{-340,-120},{-340,-160},{-322,-160}},
                                                                     color={0,0,
            127}));
    connect(minFloBypHW.y, valBypBoi.y) annotation (Line(points={{-53,-248},{-44,-248},
            {-44,-358},{178,-358},{178,-230},{230,-230},{230,-240}},
                                                     color={0,0,127}));
    connect(plaReqBoi.yPlaReq, boiPlaEnaDis.yPlaReq) annotation (Line(points={{-299,
            -160},{-290,-160},{-290,-164},{-280,-164}},      color={255,127,0}));
    connect(boiPlaEnaDis.yPla, triResHW.uDevSta) annotation (Line(points={{-257,-160},
            {-240,-160},{-240,-80},{-182,-80},{-182,-221},{-160,-221}},
                                                        color={255,0,255}));
    connect(TOut.y, boiPlaEnaDis.TOut) annotation (Line(points={{-279,180},{-260,
            180},{-260,-68},{-252,-68},{-252,-118},{-288,-118},{-288,-155.4},{
            -280,-155.4}},
          color={0,0,127}));
    connect(conAHU.ySupFan, boiPlaEnaDis.ySupFan) annotation (Line(points={{424,
            629.333},{436,629.333},{436,658},{-258,658},{-258,-110},{-292,-110},
            {-292,-160},{-280,-160}},
                          color={255,0,255}));
    connect(swiFreSta.y, proHWVal.u1) annotation (Line(points={{42,-130},{48,-130},
            {48,-156},{22,-156},{22,-174},{38,-174}}, color={0,0,127}));
    connect(proHWVal.y, HWVal.y)
      annotation (Line(points={{62,-180},{86,-180}}, color={0,0,127}));
    connect(booToReaHW.y, proHWVal.u2) annotation (Line(points={{-197,-160},{-94,-160},
            {-94,-186},{38,-186}}, color={0,0,127}));
    connect(proCHWVal.y, watVal.y)
      annotation (Line(points={{490,-108},{526,-108}}, color={0,0,127}));
    connect(booToRea.y, proCHWVal.u2) annotation (Line(points={{1189,-116},{1228,-116},
            {1228,-74},{436,-74},{436,-114},{466,-114}}, color={0,0,127}));
    connect(plaReqChi.uPlaVal, conAHU.yCoo) annotation (Line(points={{1042,-110},{
            1016,-110},{1016,-72},{388,-72},{388,44},{448,44},{448,544},{424,544}},
          color={0,0,127}));
    connect(conAHU.yCoo, proCHWVal.u1) annotation (Line(points={{424,544},{450,544},
            {450,-102},{466,-102}}, color={0,0,127}));
    connect(fanSup.y_actual, chiPlaEnaDis.yFanSpe) annotation (Line(points={{321,
            -33},{382,-33},{382,-68},{1080,-68},{1080,-117},{1099,-117}}, color={
            0,0,127}));
    connect(fanSup.y_actual, boiPlaEnaDis.yFanSpe) annotation (Line(points={{321,
            -33},{384,-33},{384,28},{16,28},{16,-66},{-256,-66},{-256,-124},{-294,
            -124},{-294,-167},{-279,-167}}, color={0,0,127}));
    connect(yVal6.y, chiWSE.yVal6) annotation (Line(points={{1039,-198},{866,-198},
            {866,-207.8},{695.6,-207.8}}, color={0,0,127}));
    connect(temDifPreRes.TSet, oveActTCHWSup.u) annotation (Line(points={{1111,-247},
            {1116,-247},{1116,-270},{1130,-270}}, color={0,0,127}));
    connect(oveActTCHWSup.y, chiWSE.TSet) annotation (Line(points={{1153,-270},{1162,
            -270},{1162,-336},{872,-336},{872,-218.8},{695.6,-218.8}}, color={0,0,
            127}));
    connect(oveActTCHWSup.y, cooTowSpeCon.TCHWSupSet) annotation (Line(points={{1153,
            -270},{1162,-270},{1162,-66},{1272,-66},{1272,-98},{1278,-98},{1278,
            -97.1111},{1284,-97.1111}},
                              color={0,0,127}));
    connect(oveActTCHWSup.y, cooModCon.TCHWSupSet) annotation (Line(points={{1153,
            -270},{1160,-270},{1160,-312},{1020,-312},{1020,-250.222},{1026,
            -250.222}},
          color={0,0,127}));
    connect(gasBoi.y, PGas.u)
      annotation (Line(points={{1245,554},{1282,554},{1282,540},{1318,540}},
                                                       color={0,0,127}));
    connect(flo.TRooAir, banDevSum.u1) annotation (Line(points={{1094.14,
            491.333},{1165.07,491.333},{1165.07,490},{1238,490}}, color={0,0,127}));
    connect(conAHU.ySupFan, booRepSupFan.u) annotation (Line(points={{424,
            629.333},{467,629.333},{467,644},{498,644}},
                                                color={255,0,255}));
    connect(booRepSupFan.y, banDevSum.uSupFan) annotation (Line(points={{522,644},
            {580,644},{580,656},{1154,656},{1154,484},{1238,484}},      color={
            255,0,255}));
    connect(eleTot.y, PHVAC.u)
      annotation (Line(points={{1301.02,704},{1318,704}}, color={0,0,127}));
    connect(TAirDev.y, dtTZonAir.u) annotation (Line(points={{1295.02,490},{
            1300,490},{1300,448},{1318,448}}, color={0,0,127}));
    connect(TOutAir.u, weaBus.TDryBul) annotation (Line(points={{-302,130},{
            -312,130},{-312,180},{-320,180}}, color={0,0,127}), Text(
        string="%second",
        index=1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(GHI.u, weaBus.HGloHor) annotation (Line(points={{-302,90},{-314,90},
            {-314,180},{-320,180}}, color={0,0,127}), Text(
        string="%second",
        index=1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(cor.y_actual, minyDam.u[1]) annotation (Line(
        points={{612,58},{616,58},{616,288.4},{1218,288.4}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(cor.y_actual, maxyDam.u[1]) annotation (Line(
        points={{612,58},{622,58},{622,250.4},{1218,250.4}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(sou.y_actual, minyDam.u[2]) annotation (Line(
        points={{792,56},{800,56},{800,54},{806,54},{806,289.2},{1218,289.2}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(sou.y_actual, maxyDam.u[2]) annotation (Line(
        points={{792,56},{830,56},{830,251.2},{1218,251.2}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(eas.y_actual, minyDam.u[3]) annotation (Line(
        points={{972,56},{994,56},{994,290},{1218,290}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(eas.y_actual, maxyDam.u[3]) annotation (Line(
        points={{972,56},{984,56},{984,252},{1218,252}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(nor.y_actual, minyDam.u[4]) annotation (Line(
        points={{1132,56},{1142,56},{1142,290},{1218,290},{1218,290.8}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(nor.y_actual, maxyDam.u[4]) annotation (Line(
        points={{1132,56},{1148,56},{1148,252.8},{1218,252.8}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(wes.y_actual, minyDam.u[5]) annotation (Line(
        points={{1332,56},{1360,56},{1360,198},{1164,198},{1164,291.6},{1218,
            291.6}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(wes.y_actual, maxyDam.u[5]) annotation (Line(
        points={{1332,56},{1356,56},{1356,204},{1170,204},{1170,253.6},{1218,
            253.6}},
        color={0,0,127},
        pattern=LinePattern.Dash));
    connect(minyDam.y, yDamMin.u)
      annotation (Line(points={{1241,290},{1258,290}}, color={0,0,127}));
    connect(maxyDam.y, yDamMax.u)
      annotation (Line(points={{1241,252},{1258,252}}, color={0,0,127}));
    connect(watVal.y_actual, yWatVal.u) annotation (Line(points={{531,-113},{
            531,-128},{450,-128},{450,-154},{466,-154}}, color={0,0,127}));
    connect(cooTow.y, yCooTowFan.u) annotation (Line(points={{736,-308},{720,
            -308},{720,-370},{738,-370}}, color={0,0,127}));
    connect(yFanSpe.u, fanSup.y) annotation (Line(points={{298,-110},{270,-110},
            {270,-16},{310,-16},{310,-28}}, color={0,0,127}));
    annotation (
      Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-400,-400},{1440,
              750}})),
      experiment(
        StartTime=17625600,
        StopTime=18230400,
        __Dymola_Algorithm="Cvode"));
  end SystemBaseline;

  model SystemCoolSeasonBaseline
    extends SystemBaseline(
        flo(
        cor(T_start=273.15 + 24),
        sou(T_start=273.15 + 24),
        eas(T_start=273.15 + 24),
        wes(T_start=273.15 + 24),
        nor(T_start=273.15 + 24)), boiPlaEnaDis(tWai=10*60));
    annotation (experiment(
        StartTime=17625600,
        StopTime=18230400,
        Tolerance=1e-06,
        __Dymola_Algorithm="Cvode"));
  end SystemCoolSeasonBaseline;

  model wrappedcool "Wrapped model for cooling case"
   // Input overwrite
   Modelica.Blocks.Interfaces.RealInput oveAct_TSupSet(unit="K", min=273.15+12, max=273.15+18) "Supply air temperature setpoint";
   Modelica.Blocks.Interfaces.RealInput oveAct_TCHWSupSet(unit="K", min=273.15+5, max=273.15+10) "Supply chilled water temperature setpoint";
   Modelica.Blocks.Interfaces.RealInput oveAct_dpSet(unit="Pa") "Supply chilled water temperature setpoint";
   // Out read
   Modelica.Blocks.Interfaces.RealOutput TZoneAirDev_y(unit="K") = modCoo.dtTZonAir.y "Total zone air temperature deviation";
   Modelica.Blocks.Interfaces.RealOutput TOutAir_y(unit="K") = modCoo.TOutAir.y "Outdoor air temperature";
   Modelica.Blocks.Interfaces.RealOutput GHI_y(unit="W/m2") = modCoo.GHI.y "Global horizontal solar radiation";
   Modelica.Blocks.Interfaces.RealOutput PHVAC_y(unit="W") = modCoo.PHVAC.y "Total HVAC power";
   Modelica.Blocks.Interfaces.RealOutput yFanSpe_y(unit="1") = modCoo.yFanSpe.y "AHU fan speed";
   Modelica.Blocks.Interfaces.RealOutput yDamMax_y(unit="1") = modCoo.yDamMax.y "Maximum zone air damper position";
   Modelica.Blocks.Interfaces.RealOutput yDamMin_y(unit="1") = modCoo.yDamMin.y "Minimum zone air damper position";
   Modelica.Blocks.Interfaces.RealOutput yWatVal_y(unit="1") = modCoo.yWatVal.y "Chilled water valve position";
   Modelica.Blocks.Interfaces.RealOutput yCooTowFan_y(unit="1") = modCoo.yCooTowFan.y "Cooling tower fan speed";
   // Original model
   FiveZone.SystemCoolSeasonBaseline modCoo(
    conAHU(supTemSetPoi(oveActTAirSup(uExt(y=oveAct_TSupSet),activate(y=true)))),
    oveActTCHWSup(uExt(y=oveAct_TCHWSupSet),activate(y=true)),
    oveActdp(uExt(y=oveAct_dpSet),activate(y=true))) "Original model with overwrites";
    annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
          coordinateSystem(preserveAspectRatio=false)));
  end wrappedcool;

  package VAVReheat "Variable air volume flow system with terminal reheat and five thermal zone"
    extends Modelica.Icons.ExamplesPackage;

    model Guideline36
      "Variable air volume flow system with terminal reheat and five thermal zones"
      extends Modelica.Icons.Example;
      extends FiveZone.VAVReheat.BaseClasses.PartialOpenLoop;

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
        final samplePeriod=samplePeriod) "Controller for terminal unit corridor"
        annotation (Placement(transformation(extent={{530,32},{550,52}})));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVSou(
        V_flow_nominal=mSou_flow_nominal/1.2,
        AFlo=AFloSou,
        final samplePeriod=samplePeriod) "Controller for terminal unit south"
        annotation (Placement(transformation(extent={{700,30},{720,50}})));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVEas(
        V_flow_nominal=mEas_flow_nominal/1.2,
        AFlo=AFloEas,
        final samplePeriod=samplePeriod) "Controller for terminal unit east"
        annotation (Placement(transformation(extent={{880,30},{900,50}})));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVNor(
        V_flow_nominal=mNor_flow_nominal/1.2,
        AFlo=AFloNor,
        final samplePeriod=samplePeriod) "Controller for terminal unit north"
        annotation (Placement(transformation(extent={{1040,30},{1060,50}})));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVWes(
        V_flow_nominal=mWes_flow_nominal/1.2,
        AFlo=AFloWes,
        final samplePeriod=samplePeriod) "Controller for terminal unit west"
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
        final nout=numZon)
        "Replicate boolean input"
        annotation (Placement(transformation(extent={{-120,280},{-100,300}})));
      Buildings.Controls.OBC.CDL.Routing.RealReplicator reaRep(
        final nout=numZon)
        "Replicate real input"
        annotation (Placement(transformation(extent={{-120,320},{-100,340}})));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.Controller conAHU(
        final pMaxSet=410,
        final yFanMin=yFanMin,
        final VPriSysMax_flow=VPriSysMax_flow,
        final peaSysPop=1.2*sum({0.05*AFlo[i] for i in 1:numZon})) "AHU controller"
        annotation (Placement(transformation(extent={{340,512},{420,640}})));
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
      connect(TSupNor.T, conVAVNor.TDis) annotation (Line(points={{1089,94},{1032,
              94},{1032,34},{1038,34}}, color={0,0,127}));
      connect(TSupWes.T, conVAVWes.TDis) annotation (Line(points={{1289,90},{1228,
              90},{1228,32},{1238,32}}, color={0,0,127}));
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
      connect(TSup.T, conVAVCor.TSupAHU) annotation (Line(points={{340,-29},{340,
              -20},{514,-20},{514,34},{528,34}},
                                            color={0,0,127}));
      connect(TSup.T, conVAVSou.TSupAHU) annotation (Line(points={{340,-29},{340,
              -20},{686,-20},{686,32},{698,32}},
                                            color={0,0,127}));
      connect(TSup.T, conVAVEas.TSupAHU) annotation (Line(points={{340,-29},{340,
              -20},{864,-20},{864,32},{878,32}},
                                            color={0,0,127}));
      connect(TSup.T, conVAVNor.TSupAHU) annotation (Line(points={{340,-29},{340,
              -20},{1028,-20},{1028,32},{1038,32}},
                                               color={0,0,127}));
      connect(TSup.T, conVAVWes.TSupAHU) annotation (Line(points={{340,-29},{340,
              -20},{1224,-20},{1224,30},{1238,30}},
                                               color={0,0,127}));
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
      connect(flo.TRooAir, TZonSet.TZon) annotation (Line(points={{1094.14,491.333},
              {1164,491.333},{1164,662},{46,662},{46,313},{58,313}}, color={0,0,127}));
      connect(occSch.occupied, booRep.u) annotation (Line(points={{-297,-216},{-160,
              -216},{-160,290},{-122,290}}, color={255,0,255}));
      connect(occSch.tNexOcc, reaRep.u) annotation (Line(points={{-297,-204},{-180,-204},
              {-180,330},{-122,330}}, color={0,0,127}));
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
              {308,589},{308,609.778},{336,609.778}},           color={0,0,127}));
      connect(zonToSys.VSumDesPopBreZon_flow, conAHU.VSumDesPopBreZon_flow)
        annotation (Line(points={{302,586},{310,586},{310,604.444},{336,604.444}},
            color={0,0,127}));
      connect(zonToSys.VSumDesAreBreZon_flow, conAHU.VSumDesAreBreZon_flow)
        annotation (Line(points={{302,583},{312,583},{312,599.111},{336,599.111}},
            color={0,0,127}));
      connect(zonToSys.yDesSysVenEff, conAHU.uDesSysVenEff) annotation (Line(points={{302,580},
              {314,580},{314,593.778},{336,593.778}},           color={0,0,127}));
      connect(zonToSys.VSumUncOutAir_flow, conAHU.VSumUncOutAir_flow) annotation (
          Line(points={{302,577},{316,577},{316,588.444},{336,588.444}}, color={0,0,
              127}));
      connect(zonToSys.VSumSysPriAir_flow, conAHU.VSumSysPriAir_flow) annotation (
          Line(points={{302,571},{318,571},{318,583.111},{336,583.111}}, color={0,0,
              127}));
      connect(zonToSys.uOutAirFra_max, conAHU.uOutAirFra_max) annotation (Line(
            points={{302,574},{320,574},{320,577.778},{336,577.778}}, color={0,0,127}));
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
            points={{424,586.667},{440,586.667},{440,468},{270,468},{270,582},{278,
              582}},
            color={0,0,127}));
      connect(conAHU.VDesUncOutAir_flow, reaRep1.u) annotation (Line(points={{424,
              597.333},{440,597.333},{440,590},{458,590}},
                                                  color={0,0,127}));
      connect(reaRep1.y, zonOutAirSet.VUncOut_flow_nominal) annotation (Line(points={{482,590},
              {490,590},{490,464},{210,464},{210,581},{218,581}},          color={0,
              0,127}));
      connect(conAHU.yReqOutAir, booRep1.u) annotation (Line(points={{424,565.333},
              {444,565.333},{444,560},{458,560}},color={255,0,255}));
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
              {140,303},{140,531.556},{336,531.556}}, color={255,127,0}));
      connect(TZonResReq.y, conAHU.uZonTemResReq) annotation (Line(points={{322,370},
              {330,370},{330,526.222},{336,526.222}}, color={255,127,0}));
      connect(PZonResReq.y, conAHU.uZonPreResReq) annotation (Line(points={{322,340},
              {326,340},{326,520.889},{336,520.889}}, color={255,127,0}));
      connect(TZonSet[1].TZonHeaSet, conAHU.TZonHeaSet) annotation (Line(points={{82,310},
              {110,310},{110,636.444},{336,636.444}},      color={0,0,127}));
      connect(TZonSet[1].TZonCooSet, conAHU.TZonCooSet) annotation (Line(points={{82,317},
              {120,317},{120,631.111},{336,631.111}},      color={0,0,127}));
      connect(TOut.y, conAHU.TOut) annotation (Line(points={{-279,180},{-260,180},{
              -260,625.778},{336,625.778}},
                                       color={0,0,127}));
      connect(dpDisSupFan.p_rel, conAHU.ducStaPre) annotation (Line(points={{311,0},
              {160,0},{160,620.444},{336,620.444}}, color={0,0,127}));
      connect(TSup.T, conAHU.TSup) annotation (Line(points={{340,-29},{340,-20},{
              152,-20},{152,567.111},{336,567.111}},
                                                 color={0,0,127}));
      connect(TRet.T, conAHU.TOutCut) annotation (Line(points={{100,151},{100,
              561.778},{336,561.778}},
                              color={0,0,127}));
      connect(VOut1.V_flow, conAHU.VOut_flow) annotation (Line(points={{-61,-20.9},
              {-61,545.778},{336,545.778}},color={0,0,127}));
      connect(TMix.T, conAHU.TMix) annotation (Line(points={{40,-29},{40,538.667},{
              336,538.667}},
                         color={0,0,127}));
      connect(conAHU.yOutDamPos, eco.yOut) annotation (Line(points={{424,522.667},{
              448,522.667},{448,36},{-10,36},{-10,-34}},
                                                     color={0,0,127}));
      connect(conAHU.yRetDamPos, eco.yRet) annotation (Line(points={{424,533.333},{
              442,533.333},{442,40},{-16.8,40},{-16.8,-34}},
                                                         color={0,0,127}));
      connect(conAHU.yCoo, gaiCooCoi.u) annotation (Line(points={{424,544},{452,544},
              {452,-274},{88,-274},{88,-248},{98,-248}}, color={0,0,127}));
      connect(conAHU.yHea, swiFreSta.u3) annotation (Line(points={{424,554.667},{
              458,554.667},{458,-280},{40,-280},{40,-200},{58,-200}},
                                                                  color={0,0,127}));
      connect(conAHU.ySupFanSpe, fanSup.y) annotation (Line(points={{424,618.667},{
              432,618.667},{432,-14},{310,-14},{310,-28}},
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
</html>",     revisions="<html>
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
        experiment(StopTime=172800, Tolerance=1e-06),
        Icon(coordinateSystem(extent={{-100,-100},{100,100}})));
    end Guideline36;

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
        import FiveZone.VAVReheat.Controls.OperationModes;
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
        import FiveZone.VAVReheat.Controls.OperationModes;
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
        import FiveZone.VAVReheat.Controls.OperationModes;
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
        import FiveZone.VAVReheat.Controls.OperationModes;
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
          mode=FiveZone.VAVReheat.Controls.OperationModes.unoccupiedOff,
          nIn=3,
          nOut=4) "Unoccupied off mode, no coil protection"
          annotation (Placement(transformation(extent={{-20,20},{0,40}})));
        State unOccNigSetBac(
          nOut=2,
          mode=FiveZone.VAVReheat.Controls.OperationModes.unoccupiedNightSetBack,
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
        State morWarUp(mode=FiveZone.VAVReheat.Controls.OperationModes.unoccupiedWarmUp,
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
        State occ(       mode=FiveZone.VAVReheat.Controls.OperationModes.occupied,
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
          mode=FiveZone.VAVReheat.Controls.OperationModes.unoccupiedPreCool,
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
        import FiveZone.VAVReheat.Controls.OperationModes;
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

      block Controller
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

        Buildings.Controls.OBC.CDL.Interfaces.RealInput TOut(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") "Outdoor air temperature"
          annotation (Placement(transformation(extent={{-240,220},{-200,260}}),
              iconTransformation(extent={{-240,260},{-200,300}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput ducStaPre(
          final unit="Pa",
          final displayUnit="Pa")
          "Measured duct static pressure"
          annotation (Placement(transformation(extent={{-240,190},{-200,230}}),
              iconTransformation(extent={{-240,230},{-200,270}})));

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

         Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.SupplyFan
                                                     supFan(
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

        FiveZone.VAVReheat.Controls.SupplyTemperature supTemSetPoi(
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

        Buildings.Controls.OBC.CDL.Interfaces.RealInput uTSupSet(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature")
          "External supply air temperature setpoint"
          annotation (Placement(transformation(extent={{-240,304},{-200,344}}),
              iconTransformation(extent={{-240,304},{-200,344}})));
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

        connect(supTemSetPoi.uTSupSet, uTSupSet) annotation (Line(points={{-2,178},{-30,
                178},{-30,324},{-220,324}}, color={0,0,127}));
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
      end Controller;

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

        Buildings.Controls.OBC.CDL.Interfaces.RealInput TOut(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") "Outdoor air temperature"
          annotation (Placement(transformation(extent={{-240,220},{-200,260}}),
              iconTransformation(extent={{-240,260},{-200,300}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput ducStaPre(
          final unit="Pa",
          final displayUnit="Pa")
          "Measured duct static pressure"
          annotation (Placement(transformation(extent={{-240,190},{-200,230}}),
              iconTransformation(extent={{-240,230},{-200,270}})));

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

        Buildings.Controls.OBC.ASHRAE.G36_PR1.AHUs.MultiZone.VAV.SetPoints.SupplyFan
          supFan(
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

        FiveZone.VAVReheat.Controls.SupplyTemperatureOve
          supTemSetPoi(
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

      block SupplyTemperature
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
          annotation (Placement(transformation(extent={{-180,-30},{-140,10}}),
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

        Buildings.Controls.OBC.CDL.Interfaces.RealInput uTSupSet(
          final unit="K",
          final displayUnit="degC",
          final quantity="ThermodynamicTemperature") "Real input signal"
          annotation (Placement(transformation(extent={{-180,-60},{-140,-20}}),
              iconTransformation(extent={{-140,-40},{-100,0}})));
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
          annotation (Placement(transformation(extent={{-40,-70},{-20,-50}})));
        Buildings.Controls.OBC.CDL.Logical.And and1
          "Check if it is in Warmup or Setback mode"
          annotation (Placement(transformation(extent={{20,-100},{40,-80}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant supTemWarUpSetBac(k=
              TSupWarUpSetBac)
          "Supply temperature setpoint under warm-up and setback mode"
          annotation (Placement(transformation(extent={{20,-130},{40,-110}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi1
          "If operation mode is setup or cool-down, setpoint shall be 35 degC"
          annotation (Placement(transformation(extent={{80,-62},{100,-42}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi2
          "If operation mode is setup or cool-down, setpoint shall be TSupSetMin"
          annotation (Placement(transformation(extent={{20,-70},{40,-50}})));
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
          annotation (Placement(transformation(extent={{-100,-70},{-80,-50}})));
        Buildings.Controls.OBC.CDL.Integers.GreaterThreshold intGreThr(
          threshold=Buildings.Controls.OBC.ASHRAE.G36_PR1.Types.OperationModes.occupied)
          "Check if operation mode index is greater than occupied mode index (1)"
          annotation (Placement(transformation(extent={{-100,-100},{-80,-80}})));
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
          annotation (Line(points={{42,-90},{60,-90},{60,-52},{78,-52}},
            color={255,0,255}));
        connect(supTemWarUpSetBac.y, swi1.u1)
          annotation (Line(points={{42,-120},{68,-120},{68,-44},{78,-44}},
            color={0,0,127}));
        connect(and2.y, swi2.u2)
          annotation (Line(points={{-18,-60},{18,-60}},color={255,0,255}));
        connect(minSupTem.y, swi2.u1)
          annotation (Line(points={{-78,-10},{0,-10},{0,-52},{18,-52}},
            color={0,0,127}));
        connect(swi2.y, swi1.u3)
          annotation (Line(points={{42,-60},{78,-60}},
            color={0,0,127}));
        connect(TZonSetAve, TDea.u)
          annotation (Line(points={{-160,90},{-102,90}},
            color={0,0,127}));
        connect(uSupFan, swi3.u2)
          annotation (Line(points={{-160,-10},{-120,-10},{-120,10},{-60,10},{-60,0},{78,
                0}}, color={255,0,255}));
        connect(swi1.y, swi3.u1)
          annotation (Line(points={{102,-52},{110,-52},{110,-20},{68,-20},{68,8},{78,8}},
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
          annotation (Line(points={{-78,-60},{-42,-60}},color={255,0,255}));
        connect(intGreThr.y, and2.u2)
          annotation (Line(points={{-78,-90},{-60,-90},{-60,-68},{-42,-68}},
            color={255,0,255}));
        connect(uOpeMod, intLesThr.u)
          annotation (Line(points={{-160,-100},{-120,-100},{-120,-60},{-102,-60}},
            color={255,127,0}));
        connect(uOpeMod, intGreThr.u)
          annotation (Line(points={{-160,-100},{-120,-100},{-120,-90},{-102,-90}},
            color={255,127,0}));
        connect(uOpeMod, intLesThr1.u)
          annotation (Line(points={{-160,-100},{-50,-100},{-50,-90},{-42,-90}},
            color={255,127,0}));
        connect(uOpeMod, intGreThr1.u)
          annotation (Line(points={{-160,-100},{-120,-100},{-120,-120},{-42,-120}},
            color={255,127,0}));
        connect(uZonTemResReq, maxSupTemRes.numOfReq)
          annotation (Line(points={{-160,20},{-112,20},{-112,22},{-102,22}},
            color={255,127,0}));
        connect(uSupFan, maxSupTemRes.uDevSta)
          annotation (Line(points={{-160,-10},{-120,-10},{-120,38},{-102,38}},
            color={255,0,255}));
        connect(maxSupTemRes.y, lin.f1)
          annotation (Line(points={{-78,30},{-60,30},{-60,54},{18,54}},
            color={0,0,127}));
        connect(swi3.y, TSupSet)
          annotation (Line(points={{102,0},{160,0}},   color={0,0,127}));

        connect(swi2.u3, uTSupSet) annotation (Line(points={{18,-68},{-10,-68},{-10,-40},
                {-160,-40}},
                     color={0,0,127}));
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
      end SupplyTemperature;

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

      block SupplyFan "Block to control multi zone VAV AHU supply fan"

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
          annotation (Placement(transformation(extent={{-200,-138},{-160,-98}}),
              iconTransformation(extent={{-140,-100},{-100,-60}})));
        Buildings.Controls.OBC.CDL.Interfaces.IntegerInput uZonPreResReq
          "Zone static pressure reset requests"
          annotation (Placement(transformation(extent={{-200,-88},{-160,-48}}),
            iconTransformation(extent={{-140,-50},{-100,-10}})));
        Buildings.Controls.OBC.CDL.Interfaces.BooleanOutput ySupFan "Supply fan on status"
          annotation (Placement(transformation(extent={{140,50},{180,90}}),
              iconTransformation(extent={{100,50},{140,90}})));
        Buildings.Controls.OBC.CDL.Interfaces.RealOutput ySupFanSpe(
          min=0,
          max=1,
          final unit="1") "Supply fan speed"
          annotation (Placement(transformation(extent={{140,-128},{180,-88}}),
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
          annotation (Placement(transformation(extent={{-130,-68},{-110,-48}})));
        Buildings.Controls.OBC.CDL.Continuous.LimPID conSpe(
          final controllerType=controllerType,
          final k=k,
          final Ti=Ti,
          final Td=Td,
          final yMax=yFanMax,
          final yMin=yFanMin,
          reset=Buildings.Controls.OBC.CDL.Types.Reset.Parameter,
          y_reset=yFanMin) "Supply fan speed control"
          annotation (Placement(transformation(extent={{-40,-88},{-20,-68}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealInput uDucStaPreSet(
          final unit="Pa",
          quantity="PressureDifference")
          "Fixed duct static pressure setpoint"
          annotation (Placement(transformation(extent={{-200,-40},{-160,0}})));
      protected
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant zerSpe(k=0)
          "Zero fan speed when it becomes OFF"
          annotation (Placement(transformation(extent={{20,-98},{40,-78}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi
          "If fan is OFF, fan speed outputs to zero"
          annotation (Placement(transformation(extent={{80,-98},{100,-118}})));
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
          annotation (Placement(transformation(extent={{-130,-108},{-110,-88}})));
        Buildings.Controls.OBC.CDL.Continuous.Division norPSet
          "Normalization for pressure set point"
          annotation (Placement(transformation(extent={{-70,-88},{-50,-68}})));
        Buildings.Controls.OBC.CDL.Continuous.Division norPMea
          "Normalization of pressure measurement"
          annotation (Placement(transformation(extent={{-70,-128},{-50,-108}})));
        Buildings.Controls.OBC.CDL.Discrete.FirstOrderHold firOrdHol(
          final samplePeriod=samplePeriod)
          "Extrapolation through the values of the last two sampled input signals"
          annotation (Placement(transformation(extent={{-100,-68},{-80,-48}})));

      equation
        connect(or2.y, or1.u2)
          annotation (Line(points={{42,40},{60,40},{60,62},{78,62}},
            color={255,0,255}));
        connect(or1.y, ySupFan)
          annotation (Line(points={{102,70},{160,70}},
            color={255,0,255}));
        connect(or1.y, staPreSetRes.uDevSta)
          annotation (Line(points={{102,70},{120,70},{120,-8},{-150,-8},{-150,-50},{-132,
                -50}},     color={255,0,255}));
        connect(or1.y, swi.u2)
          annotation (Line(points={{102,70},{120,70},{120,-8},{0,-8},{0,-108},{78,-108}},
            color={255,0,255}));
        connect(conSpe.y, swi.u1)
          annotation (Line(points={{-18,-78},{-4,-78},{-4,-116},{78,-116}},
            color={0,0,127}));
        connect(zerSpe.y, swi.u3)
          annotation (Line(points={{42,-88},{60,-88},{60,-100},{78,-100}},
            color={0,0,127}));
        connect(swi.y, ySupFanSpe)
          annotation (Line(points={{102,-108},{160,-108}},
            color={0,0,127}));
        connect(uZonPreResReq, staPreSetRes.numOfReq)
          annotation (Line(points={{-180,-68},{-148,-68},{-148,-66},{-132,-66}},
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
          annotation (Line(points={{-48,-78},{-42,-78}}, color={0,0,127}));
        connect(norPMea.y, conSpe.u_m)
          annotation (Line(points={{-48,-118},{-30,-118},{-30,-90}}, color={0,0,127}));
        connect(staPreSetRes.y, firOrdHol.u)
          annotation (Line(points={{-108,-58},{-102,-58}}, color={0,0,127}));
        connect(conSpe.trigger, or1.y)
          annotation (Line(points={{-36,-90},{-36,-100},{0,-100},{0,-8},{120,-8},{120,
                70},{102,70}},  color={255,0,255}));
        connect(gaiNor.y, norPSet.u2) annotation (Line(points={{-108,-98},{-92,-98},{-92,
                -84},{-72,-84}}, color={0,0,127}));
        connect(ducStaPre, norPMea.u1) annotation (Line(points={{-180,-118},{-80,-118},
                {-80,-112},{-72,-112}}, color={0,0,127}));
        connect(gaiNor.y, norPMea.u2) annotation (Line(points={{-108,-98},{-92,-98},{-92,
                -124},{-72,-124}}, color={0,0,127}));

        connect(norPSet.u1, uDucStaPreSet) annotation (Line(points={{-72,-72},{-74,-72},
                {-74,-20},{-180,-20}}, color={0,0,127}));
      annotation (
        defaultComponentName="conSupFan",
        Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-160,-140},{140,160}}),
              graphics={
              Rectangle(
                extent={{-156,-30},{134,-136}},
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
                extent={{54,-34},{124,-46}},
                lineColor={0,0,255},
                fillColor={215,215,215},
                fillPattern=FillPattern.Solid,
                horizontalAlignment=TextAlignment.Left,
                textString="Reset pressure setpoint"),
              Text(
                extent={{-34,-114},{20,-144}},
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
      end SupplyFan;

      package Examples "Example models to test the components"
          extends Modelica.Icons.ExamplesPackage;
        model OperationModes "Test model for operation modes"
            extends Modelica.Icons.Example;
          import ModelicaVAV = FiveZone.VAVReheat;
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

          FiveZone.VAVReheat.Controls.RoomVAV vavBoxCon
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

    package Validation "Collection of validation models"
      extends Modelica.Icons.ExamplesPackage;

      model Guideline36SteadyState
        "Validation of detailed model that is at steady state with constant weather data"
        extends FiveZone.VAVReheat.Guideline36(
          weaDat(
            pAtmSou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            ceiHeiSou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            totSkyCovSou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            opaSkyCovSou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            TDryBulSou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            TDewPoiSou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            TBlaSkySou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            TBlaSky=293.15,
            relHumSou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            winSpeSou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            winDirSou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            HInfHorSou=Buildings.BoundaryConditions.Types.DataSource.Parameter,
            HSou=Buildings.BoundaryConditions.Types.RadiationDataSource.Input_HGloHor_HDifHor),
          use_windPressure=false,
          sampleModel=false,
          flo(gai(K=0*[0.4; 0.4; 0.2])),
          occSch(occupancy=3600*24*365*{1,2}, period=2*3600*24*365));

        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant solRad(k=0) "Solar radiation"
          annotation (Placement(transformation(extent={{-400,160},{-380,180}})));
      equation
        connect(weaDat.HDifHor_in, solRad.y) annotation (Line(points={{-361,170.5},{-370,
                170.5},{-370,170},{-378,170}}, color={0,0,127}));
        connect(weaDat.HGloHor_in, solRad.y) annotation (Line(points={{-361,167},{-370,
                167},{-370,170},{-378,170}}, color={0,0,127}));
        annotation (
          experiment(
            StopTime=604800,
            Tolerance=1e-06),
            __Dymola_Commands(file="modelica://Buildings/Resources/Scripts/Dymola/Examples/VAVReheat/Validation/Guideline36SteadyState.mos"
              "Simulate and plot"),
          Diagram(coordinateSystem(extent={{-420,-300},{1360,660}})),
          Icon(coordinateSystem(extent={{-100,-100},{100,100}})),
          Documentation(info="<html>
<p>
This model validates that the detailed model of multiple rooms and an HVAC system
starts at and remains at exactly <i>20</i>&deg;C room air temperature
if there is no solar radiation, constant outdoor conditions, no internal gains and
no HVAC operation.
</p>
</html>",       revisions="<html>
<ul>
<li>
April 18, 2020, by Michael Wetter:<br/>
First implementation.
</li>
</ul>
</html>"));
      end Guideline36SteadyState;
    annotation (preferredView="info", Documentation(info="<html>
<p>
This package contains validation models for the classes in
<a href=\"modelica://Buildings.Examples.VAVReheat\">
Buildings.Examples.VAVReheat</a>.
</p>
<p>
Note that most validation models contain simple input data
which may not be realistic, but for which the correct
output can be obtained through an analytic solution.
The examples plot various outputs, which have been verified against these
solutions. These model outputs are stored as reference data and
used for continuous validation whenever models in the library change.
</p>
</html>"));
    end Validation;

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

        FiveZone.VAVReheat.ThermalZones.VAVBranch cor(
          redeclare package MediumA = MediumA,
          redeclare package MediumW = MediumW,
          m_flow_nominal=mCor_flow_nominal,
          VRoo=VRooCor,
          allowFlowReversal=allowFlowReversal)
          "Zone for core of buildings (azimuth will be neglected)"
          annotation (Placement(transformation(extent={{570,22},{610,62}})));
        FiveZone.VAVReheat.ThermalZones.VAVBranch sou(
          redeclare package MediumA = MediumA,
          redeclare package MediumW = MediumW,
          m_flow_nominal=mSou_flow_nominal,
          VRoo=VRooSou,
          allowFlowReversal=allowFlowReversal) "South-facing thermal zone"
          annotation (Placement(transformation(extent={{750,20},{790,60}})));
        FiveZone.VAVReheat.ThermalZones.VAVBranch eas(
          redeclare package MediumA = MediumA,
          redeclare package MediumW = MediumW,
          m_flow_nominal=mEas_flow_nominal,
          VRoo=VRooEas,
          allowFlowReversal=allowFlowReversal) "East-facing thermal zone"
          annotation (Placement(transformation(extent={{930,20},{970,60}})));
        FiveZone.VAVReheat.ThermalZones.VAVBranch nor(
          redeclare package MediumA = MediumA,
          redeclare package MediumW = MediumW,
          m_flow_nominal=mNor_flow_nominal,
          VRoo=VRooNor,
          allowFlowReversal=allowFlowReversal) "North-facing thermal zone"
          annotation (Placement(transformation(extent={{1090,20},{1130,60}})));
        FiveZone.VAVReheat.ThermalZones.VAVBranch wes(
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
        FiveZone.VAVReheat.BaseClasses.MixingBox eco(
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

         FiveZone.VAVReheat.BaseClasses.BandDeviationSum banDevSum[5](each
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

      block BandDeviationSumTest
        extends Modelica.Icons.Example;

        FiveZone.VAVReheat.BaseClasses.BandDeviationSum bandDevSum(uppThreshold=
             26 + 273.15, lowThreshold=22 + 273.15)
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

  package BaseClasses
    "Base classes for system-level modeling and fault injection"

    model IntegratedPrimaryLoadSide
      "Integrated water-side economizer on the load side in a primary-only chilled water system"
      extends FiveZone.BaseClasses.PartialIntegratedPrimary(final
          m_flow_nominal={m1_flow_chi_nominal,m2_flow_chi_nominal,
            m1_flow_wse_nominal,m2_flow_chi_nominal,numChi*m2_flow_chi_nominal,
            m2_flow_wse_nominal,numChi*m2_flow_chi_nominal}, rhoStd={
            Medium1.density_pTX(
                101325,
                273.15 + 4,
                Medium1.X_default),Medium2.density_pTX(
                101325,
                273.15 + 4,
                Medium2.X_default),Medium1.density_pTX(
                101325,
                273.15 + 4,
                Medium1.X_default),Medium2.density_pTX(
                101325,
                273.15 + 4,
                Medium2.X_default),Medium2.density_pTX(
                101325,
                273.15 + 4,
                Medium2.X_default),Medium2.density_pTX(
                101325,
                273.15 + 4,
                Medium2.X_default),Medium2.density_pTX(
                101325,
                273.15 + 4,
                Medium2.X_default)});

     //Dynamics
      parameter Modelica.SIunits.Time tauPump = 1
        "Time constant of fluid volume for nominal flow in pumps, used if energy or mass balance is dynamic"
         annotation (Dialog(tab = "Dynamics", group="Pump",
         enable=not energyDynamics == Modelica.Fluid.Types.Dynamics.SteadyState));
      //Pump
      parameter Integer numPum = numChi "Number of pumps"
        annotation(Dialog(group="Pump"));
      replaceable parameter Buildings.Fluid.Movers.Data.Generic perPum[numPum]
       "Performance data for the pumps"
        annotation (Dialog(group="Pump"),
              Placement(transformation(extent={{38,78},{58,98}})));
      parameter Boolean addPowerToMedium = true
        "Set to false to avoid any power (=heat and flow work) being added to medium (may give simpler equations)"
        annotation (Dialog(group="Pump"));
      parameter Modelica.SIunits.Time riseTimePump = 30
        "Rise time of the filter (time to reach 99.6 % of an opening step)"
        annotation(Dialog(tab="Dynamics", group="Filtered speed",enable=use_inputFilter));
      parameter Modelica.Blocks.Types.Init initPum = initValve
        "Type of initialization (no init/steady state/initial state/initial output)"
        annotation(Dialog(tab="Dynamics", group="Filtered speed",enable=use_inputFilter));
      parameter Real[numPum] yPum_start = fill(0,numPum)
        "Initial value of output:0-closed, 1-fully opened"
        annotation(Dialog(tab="Dynamics", group="Filtered speed",enable=use_inputFilter));
      parameter Real[numPum] yValPum_start = fill(0,numPum)
        "Initial value of output:0-closed, 1-fully opened"
        annotation(Dialog(tab="Dynamics", group="Filtered opening",enable=use_inputFilter));
      parameter Real lValPum = 0.0001
        "Valve leakage, l=Kv(y=0)/Kv(y=1)"
        annotation(Dialog(group="Pump"));
      parameter Real kFixedValPum = pum.m_flow_nominal/sqrt(pum.dpValve_nominal)
        "Flow coefficient of fixed resistance that may be in series with valve,
    k=m_flow/sqrt(dp), with unit=(kg.m)^(1/2)."
        annotation(Dialog(group="Pump"));
      Modelica.Blocks.Interfaces.RealInput yPum[numPum](
        each final unit = "1",
        each min=0,
        each max=1)
        "Constant normalized rotational speed"
        annotation (Placement(transformation(extent={{-140,-60},{-100,-20}}),
            iconTransformation(extent={{-132,-28},{-100,-60}})));
      Modelica.Blocks.Interfaces.RealOutput powPum[numPum](
        each final quantity="Power",
        each final unit="W")
        "Electrical power consumed by the pumps"
        annotation (Placement(transformation(extent={{100,-50},{120,-30}})));

      Buildings.Applications.DataCenters.ChillerCooled.Equipment.FlowMachine_y pum(
        redeclare final package Medium = Medium2,
        final p_start=p2_start,
        final T_start=T2_start,
        final X_start=X2_start,
        final C_start=C2_start,
        final C_nominal=C2_nominal,
        final m_flow_small=m2_flow_small,
        final show_T=show_T,
        final per=perPum,
        addPowerToMedium=addPowerToMedium,
        final energyDynamics=energyDynamics,
        final massDynamics=massDynamics,
        final use_inputFilter=use_inputFilter,
        final init=initPum,
        final tau=tauPump,
        final allowFlowReversal=allowFlowReversal2,
        final num=numPum,
        final m_flow_nominal=m2_flow_chi_nominal,
        dpValve_nominal=6000,
        final CvData=Buildings.Fluid.Types.CvTypes.OpPoint,
        final deltaM=deltaM2,
        final riseTimePump=riseTimePump,
        final riseTimeValve=riseTimeValve,
        final yValve_start=yValPum_start,
        final l=lValPum,
        final kFixed=kFixedValPum,
        final yPump_start=yPum_start,
        final from_dp=from_dp2,
        final homotopyInitialization=homotopyInitialization,
        final linearizeFlowResistance=linearizeFlowResistance2)
        "Pumps"
        annotation (Placement(transformation(extent={{10,-50},{-10,-30}})));

    equation
      connect(val5.port_b, pum.port_a)
        annotation (Line(points={{40,-20},{20,-20},{20,-40},{10,-40}},
                              color={0,127,255}));
      connect(pum.port_b,val6.port_a)
        annotation (Line(points={{-10,-40},{-20,-40},{-20,-20},{-40,-20}},
                                    color={0,127,255}));
      connect(yPum, pum.u)
        annotation (Line(points={{-120,-40},{-30,-40},{-30,-28},{
              18,-28},{18,-36},{12,-36}}, color={0,0,127}));
      connect(pum.P, powPum) annotation (Line(points={{-11,-36},{-14,-36},{-14,-66},
              {86,-66},{86,-40},{110,-40}},
                                          color={0,0,127}));
      annotation (Documentation(revisions="<html>
<ul>
<li>
January 12, 2019, by Michael Wetter:<br/>
Removed wrong use of <code>each</code>.
</li>
<li>
July 1, 2017, by Yangyang Fu:<br/>
First implementation.
</li>
</ul>
</html>",     info="<html>
<p>This model implements an integrated water-side economizer (WSE) on the load side of the primary-only chilled water system, as shown in the following figure. In the configuration, users can model multiple chillers with only one integrated WSE. </p>
<p align=\"center\"><img src=\"modelica://Buildings/Resources/Images/Applications/DataCenters/ChillerCooled/Equipment/IntegraredPrimaryLoadSide.png\" alt=\"image\"/> </p>
<h4>Implementation</h4>
<p>The WSE located on the load side can see the warmest return chilled water, and hence can maximize the use time of the heat exchanger. This system have three operation modes: free cooling (FC) mode, partial mechanical cooling (PMC) mode and fully mechanical cooling (FMC) mode. </p>
<p>There are 6 valves for on/off use and minimum mass flow rate control, which can be controlled in order to switch among FC, PMC and FMC mode. </p>
<ul>
<li>V1 and V2 are associated with the chiller. When the chiller is commanded to run, V1 and V2 will be open, and vice versa. Note that when the number of chillers are larger than 1, V1 and V2 are vectored models with the same dimension as the chillers. </li>
<li>V3 and V4 are associated with the WSE. When the WSE is commanded to run, V3 and V4 will be open, and vice versa. </li>
<li>V5 is for FMC only. When FMC is on, V5 is commanded to on. Otherwise, V5 is off. </li>
<li>V6 is for FC only. When FC is on, V6 is commanded to on. Otherwise, V6 is off. </li>
<li>V7 is controlled to track a minimum flowrate through the chiller. If the cooling load is very small (e.g. when the data center start to be occupied), and the flowrate through the chiller is smaller than the minimum requirement, then V7 is open, and the valve position is controlled to meet the minimum flowrate through the chiller. If the cooling load grows, V7 will eventually be fully closed. </li>
</ul>
<p>The details about how to switch among different cooling modes are shown as: </p>
<p style=\"margin-left: 30px;\">For Free Cooling (FC) Mode: </p>
<ul>
<li>V1 and V2 are closed, and V3 and V4 are open; </li>
<li>V5 is closed; </li>
<li>V6 is open; </li>
<li>V7 is closed;</li>
</ul>
<p style=\"margin-left: 30px;\">For Partially Mechanical Cooling (PMC) Mode: </p>
<ul>
<li>V1 and V2 are open, and V3 and V4 are open; </li>
<li>V5 is closed; </li>
<li>V6 is closed; </li>
<li>V7 is controlled to track a minumum flowrate through the chiller;</li>
</ul>
<p style=\"margin-left: 30px;\">For Fully Mechanical Cooling (FMC) Mode: </p>
<ul>
<li>V1 and V2 are open, and V3 and V4 are closed; </li>
<li>V5 is open; </li>
<li>V6 is closed; </li>
<li>V7 is controlled to track a minumum flowrate through the chiller;</li>
</ul>
<h4>Reference</h4>
<ul>
<li>Stein, Jeff. 2009. Waterside Economizing in Data Centers: Design and Control Considerations.<i>ASHRAE Transactions</i>, 115(2). </li>
</ul>
</html>"),     Icon(graphics={
            Polygon(
              points={{-58,40},{-58,40}},
              lineColor={0,0,0},
              fillColor={0,0,0},
              fillPattern=FillPattern.Solid),
            Polygon(
              points={{-7,-6},{9,-6},{0,3},{-7,-6}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid,
              origin={46,-45},
              rotation=90),
            Polygon(
              points={{-6,-7},{-6,9},{3,0},{-6,-7}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid,
              origin={42,-45}),
            Ellipse(
              extent={{-14,-32},{8,-54}},
              lineColor={0,0,0},
              fillPattern=FillPattern.Sphere,
              fillColor={0,128,255}),
            Polygon(
              points={{-14,-44},{-2,-54},{-2,-32},{-14,-44}},
              lineColor={0,0,0},
              pattern=LinePattern.None,
              fillPattern=FillPattern.HorizontalCylinder,
              fillColor={255,255,255}),
            Line(points={{12,6},{12,0}}, color={0,128,255}),
            Line(points={{-70,0}}, color={0,0,0}),
            Line(points={{-18,-44}}, color={0,0,0}),
            Line(points={{-18,-44},{-14,-44}}, color={0,128,255}),
            Line(points={{8,-44},{12,-44}}, color={0,128,255})}));
    end IntegratedPrimaryLoadSide;

    model PartialIntegratedPrimary
      "Integrated water-side economizer for primary-only chilled water system"
      extends
        Buildings.Applications.DataCenters.ChillerCooled.Equipment.BaseClasses.PartialChillerWSE(
        final numVal=7);

      //Parameters for the valve used in free cooling mode
      parameter Real lVal5(min=1e-10,max=1) = 0.0001
        "Valve leakage, l=Kv(y=0)/Kv(y=1)"
        annotation(Dialog(group="Two-way valve"));
      parameter Real lVal6(min=1e-10,max=1) = 0.0001
        "Valve leakage, l=Kv(y=0)/Kv(y=1)"
        annotation(Dialog(group="Two-way valve"));
      parameter Real lVal7(min=1e-10,max=1) = 0.0001
        "Valve leakage, l=Kv(y=0)/Kv(y=1)"
        annotation(Dialog(group="Two-way valve"));
      parameter Real yVal5_start(min=0,max=1) = 0
        "Initial value of output:0-closed, 1-fully opened"
        annotation(Dialog(tab="Dynamics", group="Filtered opening",
          enable=use_inputFilter));
      parameter Real yVal6_start(min=0,max=1) = 1-yVal5_start
        "Initial value of output:0-closed, 1-fully opened"
        annotation(Dialog(tab="Dynamics", group="Filtered opening",
          enable=use_inputFilter));
      parameter Real yVal7_start(min=0,max=1) = 0
        "Initial value of output:0-closed, 1-fully opened"
        annotation(Dialog(tab="Dynamics", group="Filtered opening",
          enable=use_inputFilter));
     Modelica.Blocks.Interfaces.RealInput yVal6(
       final unit = "1",
       min=0,
       max=1)
        "Actuator position for valve 6 (0: closed, 1: open)"
        annotation (Placement(
            transformation(
            extent={{-20,-20},{20,20}},
            origin={-120,-10}), iconTransformation(
            extent={{-16,-16},{16,16}},
            origin={-116,-2})));

      Modelica.Blocks.Interfaces.RealInput yVal5(
        final unit= "1",
        min=0,
        max=1)
        "Actuator position for valve 5(0: closed, 1: open)"
        annotation (Placement(
            transformation(
            extent={{-20,-20},{20,20}},
            origin={-120,20}), iconTransformation(
            extent={{16,16},{-16,-16}},
            rotation=180,
            origin={-116,30})));

      Buildings.Fluid.Actuators.Valves.TwoWayLinear val5(
        redeclare final package Medium = Medium2,
        final CvData=Buildings.Fluid.Types.CvTypes.OpPoint,
        final allowFlowReversal=allowFlowReversal2,
        final m_flow_nominal=numChi*m2_flow_chi_nominal,
        final show_T=show_T,
        final from_dp=from_dp2,
        final homotopyInitialization=homotopyInitialization,
        final linearized=linearizeFlowResistance2,
        final deltaM=deltaM2,
        final use_inputFilter=use_inputFilter,
        final riseTime=riseTimeValve,
        final init=initValve,
        final dpFixed_nominal=0,
        final dpValve_nominal=dpValve_nominal[5],
        final l=lVal5,
        final kFixed=0,
        final rhoStd=rhoStd[5],
        final y_start=yVal5_start)
        "Bypass valve: closed when fully mechanic cooling is activated;
    open when fully mechanic cooling is activated"
        annotation (Placement(transformation(extent={{60,-30},{40,-10}})));
      Buildings.Fluid.Actuators.Valves.TwoWayLinear val6(
        redeclare final package Medium = Medium2,
        final CvData=Buildings.Fluid.Types.CvTypes.OpPoint,
        final m_flow_nominal=m2_flow_wse_nominal,
        final allowFlowReversal=allowFlowReversal2,
        final show_T=show_T,
        final from_dp=from_dp2,
        final homotopyInitialization=homotopyInitialization,
        final linearized=linearizeFlowResistance2,
        final deltaM=deltaM2,
        final use_inputFilter=use_inputFilter,
        final riseTime=riseTimeValve,
        final init=initValve,
        final dpFixed_nominal=0,
        final dpValve_nominal=dpValve_nominal[6],
        final l=lVal6,
        final kFixed=0,
        final rhoStd=rhoStd[6],
        final y_start=yVal6_start)
        "Bypass valve: closed when free cooling mode is deactivated;
    open when free cooling is activated"
        annotation (Placement(transformation(extent={{-40,-30},{-60,-10}})));

      Buildings.Fluid.Sensors.MassFlowRate bypFlo(redeclare package Medium =
            Medium2)
        "Bypass water mass flowrate"
    annotation (Placement(transformation(extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={-80,2})));
      Modelica.Blocks.Interfaces.RealOutput mCHW_flow "Chiller mass flow rate"
        annotation (Placement(transformation(extent={{100,-30},{120,-10}}),
            iconTransformation(extent={{100,-30},{120,-10}})));
      Buildings.Fluid.Actuators.Valves.TwoWayLinear val7(
        redeclare final package Medium = Medium2,
        final CvData=Buildings.Fluid.Types.CvTypes.OpPoint,
        final allowFlowReversal=allowFlowReversal2,
        final m_flow_nominal=numChi*m2_flow_chi_nominal,
        final show_T=show_T,
        final from_dp=from_dp2,
        final homotopyInitialization=homotopyInitialization,
        final linearized=linearizeFlowResistance2,
        final deltaM=deltaM2,
        final use_inputFilter=use_inputFilter,
        final riseTime=riseTimeValve,
        final init=initValve,
        final dpFixed_nominal=0,
        final dpValve_nominal=dpValve_nominal[7],
        final l=lVal7,
        final kFixed=0,
        final rhoStd=rhoStd[7],
        final y_start=yVal7_start)
        "Bypass valve: closed when fully mechanic cooling is activated;
    open when fully mechanic cooling is activated"
        annotation (Placement(transformation(extent={{10,-90},{-10,-70}})));
     Modelica.Blocks.Interfaces.RealInput yVal7(
        final unit="1",
        min=0,
        max=1) "Actuator position for valve 7 (0: closed, 1: open)"
        annotation (Placement(
            transformation(
            extent={{-20,-20},{20,20}},
            origin={-120,-80}), iconTransformation(
            extent={{-16,-16},{16,16}},
            origin={-116,-76})));
    equation
      connect(port_a2,val5. port_a)
        annotation (Line(points={{100,-60},{80,-60},{80,-20},{60,-20}},
          color={0,127,255}));
      connect(port_a2, wse.port_a2)
        annotation (Line(points={{100,-60},{88,-60},{80,-60},{80,24},{60,24}},
          color={0,127,255}));
      connect(val6.port_a, chiPar.port_a2)
        annotation (Line(points={{-40,-20},{-20,-20},{-20,24},{-40,24}},
          color={0,127,255}));
      connect(val6.port_b, port_b2)
        annotation (Line(points={{-60,-20},{-80,-20},{-80,-60},{-100,-60}},
          color={0,127,255}));
      connect(val5.y, yVal5)
        annotation (Line(points={{50,-8},{50,16},{-94,16},{-94,20},{-120,20}},
          color={0,0,127}));
      connect(yVal6, val6.y)
        annotation (Line(points={{-120,-10},{-90,-10},{-90,16},{-50,16},{-50,-8}},
          color={0,0,127}));
      connect(senTem.port_b, val5.port_b)
        annotation (Line(points={{8,24},{0,24},{0,-20},{40,-20}},
          color={0,127,255}));
      connect(bypFlo.port_a, chiPar.port_b2)
        annotation (Line(points={{-80,12},{-80,24},{-60,24}}, color={0,127,255}));
      connect(bypFlo.port_b, port_b2) annotation (Line(points={{-80,-8},{-80,-60},{
              -100,-60}}, color={0,127,255}));
      connect(bypFlo.m_flow, mCHW_flow) annotation (Line(points={{-69,2},{90,2},{90,
              -20},{110,-20}}, color={0,0,127}));
      connect(port_b2, val7.port_b) annotation (Line(points={{-100,-60},{-40,-60},{
              -40,-80},{-10,-80}}, color={0,127,255}));
      connect(yVal7, val7.y) annotation (Line(points={{-120,-80},{-40,-80},{-40,-60},
              {0,-60},{0,-68}}, color={0,0,127}));
      connect(val7.port_a, val5.port_b) annotation (Line(points={{10,-80},{20,-80},
              {20,-20},{40,-20}}, color={0,127,255}));
      annotation (Documentation(info="<html>
<p>
Partial model that implements integrated waterside economizer in primary-ony chilled water system.
</p>
</html>",     revisions="<html>
<ul>
<li>
July 1, 2017, by Yangyang Fu:<br/>
First implementation.
</li>
</ul>
</html>"),    Icon(graphics={
            Rectangle(
              extent={{32,42},{34,36}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{30,42},{32,36}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{30,4},{32,-2}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{54,42},{56,36}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{56,42},{58,36}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Polygon(
              points={{-7,-6},{9,-6},{0,3},{-7,-6}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid,
              origin={-42,-45},
              rotation=90),
            Polygon(
              points={{-6,-7},{-6,9},{3,0},{-6,-7}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid,
              origin={-46,-45}),
            Polygon(
              points={{-7,-6},{9,-6},{0,3},{-7,-6}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid,
              origin={46,-45},
              rotation=90),
            Polygon(
              points={{-6,-7},{-6,9},{3,0},{-6,-7}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid,
              origin={42,-45}),
            Line(points={{90,-60},{78,-60},{78,-44},{52,-44}}, color={0,128,255}),
            Line(points={{36,-44},{12,-44}},color={0,128,255}),
            Line(points={{-18,-44},{-36,-44}}, color={0,128,255}),
            Line(points={{-94,-60},{-78,-60},{-78,-44},{-52,-44}}, color={0,128,255}),
            Line(points={{78,-44},{78,0},{64,0}}, color={0,128,255}),
            Line(points={{24,0},{14,0},{12,0},{12,-44}}, color={0,128,255}),
            Line(points={{12,6},{12,0}}, color={0,128,255}),
            Line(points={{-70,0}}, color={0,0,0}),
            Line(points={{-72,0},{-78,0},{-78,-54}}, color={0,128,255}),
            Line(points={{-24,0},{-18,0},{-18,-44}}, color={0,128,255})}));
    end PartialIntegratedPrimary;

    partial model PartialWaterside
      "Partial model that implements water-side cooling system"
      package MediumA = Buildings.Media.Air "Medium model for air";
      package MediumW = Buildings.Media.Water "Medium model for water";

      // Chiller parameters
      parameter Integer numChi=1 "Number of chillers";
      parameter Modelica.SIunits.MassFlowRate m1_flow_chi_nominal= -QEva_nominal*(1+1/COP_nominal)/4200/6.5
        "Nominal mass flow rate at condenser water in the chillers";
      parameter Modelica.SIunits.MassFlowRate m2_flow_chi_nominal= QEva_nominal/4200/(5.56-11.56)
        "Nominal mass flow rate at evaporator water in the chillers";
      parameter Modelica.SIunits.PressureDifference dp1_chi_nominal = 46.2*1000
        "Nominal pressure";
      parameter Modelica.SIunits.PressureDifference dp2_chi_nominal = 44.8*1000
        "Nominal pressure";
        parameter Modelica.SIunits.Power QEva_nominal
        "Nominal cooling capaciaty(Negative means cooling)";
     // WSE parameters
      parameter Modelica.SIunits.MassFlowRate m1_flow_wse_nominal= m1_flow_chi_nominal
        "Nominal mass flow rate at condenser water in the chillers";
      parameter Modelica.SIunits.MassFlowRate m2_flow_wse_nominal= m2_flow_chi_nominal
        "Nominal mass flow rate at condenser water in the chillers";
      parameter Modelica.SIunits.PressureDifference dp1_wse_nominal = 33.1*1000
        "Nominal pressure";
      parameter Modelica.SIunits.PressureDifference dp2_wse_nominal = 34.5*1000
        "Nominal pressure";
      parameter Real COP_nominal=5.9 "COP";
      parameter FiveZone.Data.Chiller[numChi] perChi(
        each QEva_flow_nominal=QEva_nominal,
        each COP_nominal=COP_nominal,
        each mEva_flow_nominal=m2_flow_chi_nominal,
        each mCon_flow_nominal=m1_flow_chi_nominal);

      parameter Buildings.Fluid.Movers.Data.Generic perPumCW(
        each pressure=
              Buildings.Fluid.Movers.BaseClasses.Characteristics.flowParameters(
              V_flow=m1_flow_chi_nominal/1000*{0.2,0.6,1.0,1.2},
              dp=(dp1_chi_nominal+133500+30000+6000)*{1.2,1.1,1.0,0.6}))
        "Performance data for condenser water pumps";

      // Set point
      parameter Modelica.SIunits.Temperature TCHWSet = 273.15 + 6
        "Chilled water temperature setpoint";
      parameter Modelica.SIunits.Pressure dpSetPoi = 36000
        "Differential pressure setpoint at cooling coil";

      FiveZone.Controls.ChillerStage chiStaCon(tWai=0)
        "Chiller staging control" annotation (Placement(transformation(extent={
                {1286,-138},{1306,-118}})));
      Modelica.Blocks.Math.RealToBoolean chiOn "Real value to boolean value"
        annotation (Placement(transformation(extent={{1326,-138},{1346,-118}})));
      Modelica.Blocks.Math.IntegerToBoolean intToBoo(
        threshold=Integer(FiveZone.Types.CoolingModes.FullMechanical))
        "Inverse on/off signal for the WSE"
        annotation (Placement(transformation(extent={{1286,-164},{1306,-144}})));
      Modelica.Blocks.Logical.Not wseOn "True: WSE is on; False: WSE is off "
        annotation (Placement(transformation(extent={{1326,-164},{1346,-144}})));
      FiveZone.Controls.ConstantSpeedPumpStage CWPumCon(tWai=0)
        "Condenser water pump controller" annotation (Placement(transformation(
              extent={{1284,-216},{1304,-196}})));
      Modelica.Blocks.Sources.IntegerExpression chiNumOn(
        y=integer(chiStaCon.y))
        "The number of running chillers"
        annotation (Placement(transformation(extent={{1196,-222},{1218,-200}})));
      Modelica.Blocks.Math.Gain gai(each k=1)
                                             "Gain effect"
        annotation (Placement(transformation(extent={{1326,-216},{1346,-196}})));
      FiveZone.Controls.CoolingTowerSpeed cooTowSpeCon(
        controllerType=Modelica.Blocks.Types.SimpleController.PI,
        yMin=0,
        Ti=60,
        k=0.1) "Cooling tower speed control"
        annotation (Placement(transformation(extent={{1286,-106},{1306,-90}})));
      Modelica.Blocks.Sources.RealExpression TCWSupSet(
         y=min(29.44 + 273.15, max(273.15+ 15.56, cooTow.TAir + 3)))
        "Condenser water supply temperature setpoint"
        annotation (Placement(transformation(extent={{1196,-100},{1216,-80}})));
      replaceable Buildings.Applications.DataCenters.ChillerCooled.Equipment.BaseClasses.PartialChillerWSE chiWSE(
        redeclare replaceable package Medium1 = MediumW,
        redeclare replaceable package Medium2 = MediumW,
        numChi=numChi,
        m1_flow_chi_nominal=m1_flow_chi_nominal,
        m2_flow_chi_nominal=m2_flow_chi_nominal,
        m1_flow_wse_nominal=m1_flow_wse_nominal,
        m2_flow_wse_nominal=m2_flow_wse_nominal,
        dp1_chi_nominal=dp1_chi_nominal,
        dp1_wse_nominal=dp1_wse_nominal,
        dp2_chi_nominal=dp2_chi_nominal,
        dp2_wse_nominal=dp2_wse_nominal,
        perChi = perChi,
        use_inputFilter=false,
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        use_controller=false)
        "Chillers and waterside economizer"
        annotation (Placement(transformation(extent={{694,-198},{674,-218}})));
      Buildings.Fluid.Sources.Boundary_pT expVesCW(redeclare replaceable
          package Medium =
                   MediumW, nPorts=1)
        "Expansion tank"
        annotation (Placement(transformation(extent={{-9,-9.5},{9,9.5}},
            rotation=180,
            origin={969,-299.5})));
      Buildings.Fluid.HeatExchangers.CoolingTowers.Merkel   cooTow(
        redeclare each replaceable package Medium = MediumW,
        ratWatAir_nominal=1.5,
        each TAirInWB_nominal(displayUnit="degC") = 273.15 + 25.55,
        each energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyStateInitial,
        each dp_nominal=30000,
        each m_flow_nominal=m1_flow_chi_nominal,
        TWatIn_nominal=273.15 + 35,
        TWatOut_nominal=((273.15 + 35) - 273.15 - 5.56) + 273.15,
        each PFan_nominal=4300)  "Cooling tower" annotation (Placement(
            transformation(extent={{-10,-10},{10,10}}, origin={748,-316})));
      Buildings.Fluid.Sensors.TemperatureTwoPort TCHWSup(redeclare replaceable
          package Medium = MediumW, m_flow_nominal=numChi*m2_flow_chi_nominal)
        "Chilled water supply temperature"
        annotation (Placement(transformation(extent={{778,-138},{758,-118}})));
      Buildings.Fluid.Sensors.TemperatureTwoPort TCWSup(redeclare replaceable
          package Medium = MediumW, m_flow_nominal=numChi*m1_flow_chi_nominal)
        "Condenser water supply temperature"
        annotation (Placement(transformation(extent={{818,-326},{838,-306}})));
      Buildings.Fluid.Sensors.TemperatureTwoPort TCWRet(redeclare replaceable
          package Medium = MediumW, m_flow_nominal=numChi*m1_flow_chi_nominal)
        "Condenser water return temperature"
        annotation (Placement(transformation(extent={{534,-326},{554,-306}})));
      Buildings.Fluid.Movers.SpeedControlled_y     pumCW(
        redeclare each replaceable package Medium = MediumW,
        addPowerToMedium=false,
        per=perPumCW,
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial)
                                    "Condenser water pump" annotation (Placement(
            transformation(
            extent={{10,10},{-10,-10}},
            rotation=-90,
            origin={910,-288})));
      Buildings.Fluid.Sensors.TemperatureTwoPort TCHWRet(redeclare replaceable
          package Medium = MediumW, m_flow_nominal=numChi*m2_flow_chi_nominal)
        "Chilled water return temperature"
        annotation (Placement(transformation(extent={{618,-198},{598,-178}})));
      Buildings.Fluid.Sources.Boundary_pT expVesChi(redeclare replaceable
          package Medium =
                   MediumW, nPorts=1)
        "Expansion tank"
        annotation (Placement(transformation(extent={{10,-10},{-10,10}},
            rotation=180,
            origin={512,-179})));
      Buildings.Fluid.Sensors.RelativePressure senRelPre(redeclare replaceable
          package Medium = MediumW)
        "Differential pressure"
        annotation (Placement(transformation(extent={{578,-130},{558,-150}})));
      Buildings.Fluid.Actuators.Valves.TwoWayLinear val(
        redeclare each package Medium = MediumW,
        m_flow_nominal=m1_flow_chi_nominal,
        dpValve_nominal=6000,
        dpFixed_nominal=133500) "Shutoff valves"
        annotation (Placement(transformation(extent={{636,-326},{656,-306}})));
      Buildings.Controls.Continuous.LimPID pumSpe(
        controllerType=Modelica.Blocks.Types.SimpleController.PI,
        Ti=40,
        yMin=0.2,
        k=0.1,
        reset=Buildings.Types.Reset.Parameter,
        y_reset=0)
               "Pump speed controller"
        annotation (Placement(transformation(extent={{1340,-258},{1360,-238}})));
      Modelica.Blocks.Math.Gain dpGai(k=1/dpSetPoi) "Gain effect"
        annotation (Placement(transformation(extent={{1256,-292},{1276,-272}})));
      Buildings.Fluid.Actuators.Valves.TwoWayEqualPercentage     watVal
        "Two-way valve"
         annotation (
          Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=270,
            origin={538,-108})));
      Buildings.Fluid.FixedResistances.PressureDrop resCHW(
        m_flow_nominal=m2_flow_chi_nominal,
        redeclare package Medium = MediumW,
        dp_nominal=150000) "Resistance in chilled water loop"
        annotation (Placement(transformation(extent={{630,-198},{650,-178}})));
      FiveZone.Controls.TemperatureDifferentialPressureReset temDifPreRes(
        dpMin(displayUnit="Pa"),
        dpMax(displayUnit="Pa"),
        TMin(displayUnit="K"),
        TMax(displayUnit="K")) annotation (Placement(transformation(extent={{1090,
                -252},{1110,-232}})));
      Modelica.Blocks.Math.Gain dpSetGai(k=1/dpSetPoi) "Gain effect"
        annotation (Placement(transformation(extent={{1256,-258},{1276,-238}})));
      Buildings.Utilities.IO.SignalExchange.Overwrite oveActdp(description=
            "chilled water supply dp setpoint", u(unit="Pa"))
        "Overwrite the chilled water dp setpoint" annotation (Placement(
            transformation(extent={{1130,-240},{1150,-220}})));
    equation

      connect(chiStaCon.y,chiOn. u)
        annotation (Line(
          points={{1307,-128},{1324,-128}},
          color={0,0,127}));
      connect(intToBoo.y,wseOn. u)
        annotation (Line(
          points={{1307,-154},{1324,-154}},
          color={255,0,255}));
      connect(TCWSupSet.y,cooTowSpeCon. TCWSupSet)
        annotation (Line(
          points={{1217,-90},{1284,-90}},
          color={0,0,127}));
      connect(chiNumOn.y,CWPumCon. numOnChi)
        annotation (Line(
          points={{1219.1,-211},{1282,-211}},
          color={255,127,0}));
      connect(dpGai.y, pumSpe.u_m) annotation (Line(points={{1277,-282},{1350,-282},
              {1350,-260}},                     color={0,0,127}));

      connect(val.port_b,cooTow. port_a)
        annotation (Line(points={{656,-316},{738,-316}}, color={0,0,255},
          thickness=0.5));
      connect(TCWSup.port_b, expVesCW.ports[1]) annotation (Line(points={{838,-316},
              {938,-316},{938,-299.5},{960,-299.5}}, color={0,0,255},
          thickness=0.5));
      connect(senRelPre.p_rel, dpGai.u) annotation (Line(points={{568,-131},{568,
              -18},{1182,-18},{1182,-282},{1254,-282}},
                                                     color={0,0,127}));
      connect(CWPumCon.y[1], gai.u) annotation (Line(points={{1305,-206.5},{1306,
              -206.5},{1306,-206},{1324,-206}},
                                        color={0,0,127}));
      connect(chiWSE.port_a1, pumCW.port_b) annotation (Line(points={{694,-214},{
              708,-214},{708,-228},{910,-228},{910,-278}},
                                     color={0,0,255},
          thickness=0.5));
      connect(TCWSup.port_b, pumCW.port_a) annotation (Line(points={{838,-316},{910,
              -316},{910,-298}}, color={0,0,255},
          thickness=0.5));
      connect(cooTow.port_b, TCWSup.port_a)
        annotation (Line(points={{758,-316},{818,-316}}, color={0,0,255},
          thickness=0.5));
      connect(TCWRet.port_b, val.port_a)
        annotation (Line(points={{554,-316},{636,-316}}, color={0,0,255},
          thickness=0.5));
      connect(dpSetGai.y, pumSpe.u_s)
        annotation (Line(points={{1277,-248},{1338,-248}}, color={0,0,127}));
      connect(chiWSE.port_b2,TCHWSup. port_a)
        annotation (Line(
          points={{694,-202},{718,-202},{718,-188},{906,-188},{906,-128},{778,-128}},
          color={28,108,200},
          thickness=0.5));
      connect(senRelPre.port_b, TCHWRet.port_b) annotation (Line(points={{558,-140},
              {538,-140},{538,-188},{598,-188}}, color={28,108,200},
          thickness=0.5));
      connect(TCWRet.port_a,chiWSE. port_b1) annotation (Line(points={{534,-316},{
              502,-316},{502,-228},{664,-228},{664,-214},{674,-214}},
                                            color={0,0,255},
          thickness=0.5));
      connect(watVal.port_b, TCHWRet.port_b) annotation (Line(points={{538,-118},{
              538,-188},{598,-188}}, color={28,108,200},
          thickness=0.5));
      connect(expVesChi.ports[1], TCHWRet.port_b) annotation (Line(points={{522,
              -179},{538,-179},{538,-188},{598,-188}}, color={28,108,200},
          thickness=0.5));
      connect(senRelPre.port_a, TCHWSup.port_b) annotation (Line(points={{578,-140},
              {594,-140},{594,-128},{758,-128}}, color={28,108,200},
          thickness=0.5));
      connect(TCHWRet.port_a, resCHW.port_a)
        annotation (Line(points={{618,-188},{630,-188}}, color={28,108,200},
          thickness=0.5));
      connect(resCHW.port_b, chiWSE.port_a2) annotation (Line(points={{650,-188},{
              662,-188},{662,-202},{674,-202}}, color={28,108,200},
          thickness=0.5));
      connect(temDifPreRes.dpSet, oveActdp.u) annotation (Line(points={{1111,
              -237},{1121.5,-237},{1121.5,-230},{1128,-230}}, color={0,0,127}));
      connect(oveActdp.y, dpSetGai.u) annotation (Line(points={{1151,-230},{
              1220,-230},{1220,-248},{1254,-248}}, color={0,0,127}));
      annotation (Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-400,
                -500},{650,20}})), Documentation(info="<html>
<p>
This is a partial model that describes the chilled water cooling system in a data center. The sizing data
are collected from the reference.
</p>
<h4>Reference </h4>
<ul>
<li>
Taylor, S. T. (2014). How to design &amp; control waterside economizers. ASHRAE Journal, 56(6), 30-36.
</li>
</ul>
</html>",     revisions="<html>
<ul>
<li>
January 12, 2019, by Michael Wetter:<br/>
Removed wrong <code>each</code>.
</li>
<li>
December 1, 2017, by Yangyang Fu:<br/>
Used scaled differential pressure to control the speed of pumps. This can avoid retuning gains
in PID when changing the differential pressure setpoint.
</li>
<li>
September 2, 2017, by Michael Wetter:<br/>
Changed expansion vessel to use the more efficient implementation.
</li>
<li>
July 30, 2017, by Yangyang Fu:<br/>
First implementation.
</li>
</ul>
</html>"));
    end PartialWaterside;

    partial model PartialHotWaterside
      "Partial model that implements hot water-side system"
      package MediumA = Buildings.Media.Air "Medium model for air";
      package MediumW = Buildings.Media.Water "Medium model for water";

      // Boiler parameters
      parameter Modelica.SIunits.MassFlowRate m_flow_boi_nominal= Q_flow_boi_nominal/4200/5
        "Nominal water mass flow rate at boiler";
      parameter Modelica.SIunits.Power Q_flow_boi_nominal
        "Nominal heating capaciaty(Positive means heating)";
      parameter Modelica.SIunits.Pressure dpSetPoiHW = 36000
        "Differential pressure setpoint at heating coil";
      parameter Buildings.Fluid.Movers.Data.Generic perPumHW(
              pressure=Buildings.Fluid.Movers.BaseClasses.Characteristics.flowParameters(
              V_flow=m_flow_boi_nominal/1000*{0.2,0.6,1.0,1.2},
              dp=(85000+60000+6000+6000)*{1.5,1.3,1.0,0.6}))
        "Performance data for primary pumps";

      Buildings.Fluid.Actuators.Valves.TwoWayEqualPercentage HWVal(
        redeclare package Medium = MediumW,
        m_flow_nominal=m_flow_boi_nominal,
        dpValve_nominal=6000) "Two-way valve"
        annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=270,
            origin={98,-180})));
      Buildings.Fluid.Sources.Boundary_pT expVesBoi(redeclare replaceable
          package Medium =
                   MediumW,
        T=318.15,           nPorts=1)
        "Expansion tank"
        annotation (Placement(transformation(extent={{10,-10},{-10,10}},
            rotation=180,
            origin={58,-319})));
      Buildings.Fluid.Sensors.TemperatureTwoPort THWRet(redeclare replaceable
          package Medium = MediumW, m_flow_nominal=m_flow_boi_nominal)
        "Boiler plant water return temperature"
        annotation (Placement(transformation(extent={{10,-10},{-10,10}},
            rotation=90,
            origin={98,-226})));
      Buildings.Fluid.FixedResistances.PressureDrop resHW(
        m_flow_nominal=m_flow_boi_nominal,
        redeclare package Medium = MediumW,
        dp_nominal=85000)  "Resistance in hot water loop" annotation (Placement(
            transformation(
            extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={350,-260})));
      Buildings.Fluid.Boilers.BoilerPolynomial boi(
        redeclare package Medium = MediumW,
        m_flow_nominal=m_flow_boi_nominal,
        dp_nominal=60000,
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        Q_flow_nominal=Q_flow_boi_nominal,
        T_nominal=318.15,
        fue=Buildings.Fluid.Data.Fuels.NaturalGasLowerHeatingValue())
        annotation (Placement(transformation(extent={{130,-330},{150,-310}})));
      Buildings.Fluid.Actuators.Valves.TwoWayLinear boiIsoVal(
        redeclare each package Medium = MediumW,
        m_flow_nominal=m_flow_boi_nominal,
        dpValve_nominal=6000) "Boiler Isolation Valve"
        annotation (Placement(transformation(extent={{282,-330},{302,-310}})));
      Buildings.Fluid.Movers.SpeedControlled_y pumHW(
        redeclare package Medium = MediumW,
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        per=perPumHW,
        addPowerToMedium=false)
        annotation (Placement(transformation(extent={{198,-330},{218,-310}})));
      Buildings.Fluid.Sensors.TemperatureTwoPort THWSup(redeclare replaceable
          package Medium = MediumW, m_flow_nominal=m_flow_boi_nominal)
        "Hot water supply temperature" annotation (Placement(transformation(
            extent={{10,-10},{-10,10}},
            rotation=90,
            origin={350,-224})));
      Buildings.Fluid.Actuators.Valves.TwoWayEqualPercentage valBypBoi(
        redeclare package Medium = MediumW,
        m_flow_nominal=m_flow_boi_nominal,
        dpValve_nominal=6000,
        y_start=0,
        use_inputFilter=false,
        from_dp=true) "Bypass valve for boiler." annotation (Placement(
            transformation(extent={{-10,-10},{10,10}}, origin={230,-252})));
      Buildings.Fluid.Sensors.RelativePressure senRelPreHW(redeclare
          replaceable package Medium =
                           MediumW) "Differential pressure"
        annotation (Placement(transformation(extent={{208,-196},{188,-216}})));

      Modelica.Blocks.Math.Gain dpSetGaiHW(k=1/dpSetPoiHW) "Gain effect"
        annotation (Placement(transformation(extent={{-120,-310},{-100,-290}})));
      Modelica.Blocks.Math.Gain dpGaiHW(k=1/dpSetPoiHW) "Gain effect"
        annotation (Placement(transformation(extent={{-120,-350},{-100,-330}})));
      Buildings.Controls.Continuous.LimPID pumSpeHW(
        controllerType=Modelica.Blocks.Types.SimpleController.PI,
        Ti=40,
        yMin=0.2,
        k=0.1) "Pump speed controller"
        annotation (Placement(transformation(extent={{-70,-330},{-50,-310}})));
      Buildings.Controls.OBC.CDL.Continuous.Product proPumHW
        annotation (Placement(transformation(extent={{-32,-324},{-12,-304}})));
      FiveZone.Controls.MinimumFlowBypassValve minFloBypHW(controllerType=
            Modelica.Blocks.Types.SimpleController.PI)
        "Hot water loop minimum bypass valve control"
        annotation (Placement(transformation(extent={{-74,-258},{-54,-238}})));
      Buildings.Fluid.Sensors.MassFlowRate   senHWFlo(redeclare package Medium =
            MediumW)
        "Sensor for hot water flow rate" annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={98,-278})));
      Buildings.Controls.Continuous.LimPID boiTSup(
        Td=1,
        k=0.5,
        controllerType=Modelica.Blocks.Types.SimpleController.PI,
        Ti=100) "Boiler supply water temperature"
        annotation (Placement(transformation(extent={{-74,-288},{-54,-268}})));
      Buildings.Controls.OBC.CDL.Continuous.Product proBoi
        annotation (Placement(transformation(extent={{-32,-282},{-12,-262}})));
      FiveZone.Controls.TrimResponse triResHW(
        uTri=0.9,
        dpMin(displayUnit="kPa") = 0.5*dpSetPoiHW,
        dpMax(displayUnit="kPa") = dpSetPoiHW) annotation (Placement(
            transformation(extent={{-156,-236},{-136,-216}})));
    equation
      connect(HWVal.port_b, THWRet.port_a)
        annotation (Line(points={{98,-190},{98,-216}},
                                                     color={238,46,47},
          thickness=0.5));
      connect(boi.port_b, pumHW.port_a)
        annotation (Line(points={{150,-320},{198,-320}},color={238,46,47},
          thickness=0.5));
      connect(pumHW.port_b, boiIsoVal.port_a)
        annotation (Line(points={{218,-320},{282,-320}}, color={238,46,47},
          thickness=0.5));
      connect(THWRet.port_a, senRelPreHW.port_b)
        annotation (Line(points={{98,-216},{98,-206},{188,-206}},
                                                      color={238,46,47},
          thickness=0.5));
      connect(senRelPreHW.port_a, THWSup.port_a)
        annotation (Line(points={{208,-206},{350,-206},{350,-214}},
                                                       color={238,46,47},
          thickness=0.5));
      connect(dpSetGaiHW.y, pumSpeHW.u_s)
        annotation (Line(points={{-99,-300},{-86,-300},{-86,-320},{-72,-320}},
                                                           color={0,0,127}));
      connect(dpGaiHW.y, pumSpeHW.u_m) annotation (Line(points={{-99,-340},{-60,
              -340},{-60,-332}},
                            color={0,0,127}));
      connect(senRelPreHW.p_rel, dpGaiHW.u) annotation (Line(points={{198,-197},{
              198,-68},{-182,-68},{-182,-340},{-122,-340}},
                                                      color={0,0,127}));
      connect(pumSpeHW.y, proPumHW.u2)
        annotation (Line(points={{-49,-320},{-34,-320}}, color={0,0,127}));
      connect(proPumHW.y, pumHW.y) annotation (Line(points={{-10,-314},{10,-314},{
              10,-360},{180,-360},{180,-300},{208,-300},{208,-308}},
                                               color={0,0,127}));
      connect(THWSup.port_b, resHW.port_a)
        annotation (Line(points={{350,-234},{350,-250}},
                                                       color={238,46,47},
          thickness=1));
      connect(resHW.port_b, boiIsoVal.port_b) annotation (Line(points={{350,-270},{
              350,-320},{302,-320}},       color={238,46,47},
          thickness=0.5));
      connect(expVesBoi.ports[1], boi.port_a) annotation (Line(points={{68,-319},{
              100,-319},{100,-320},{130,-320}},
                                          color={238,46,47},
          thickness=0.5));
      connect(boiTSup.u_m, boi.T) annotation (Line(points={{-64,-290},{-64,-296},{
              -98,-296},{-98,-64},{160,-64},{160,-312},{151,-312}},
                                             color={0,0,127}));
      connect(senHWFlo.m_flow, minFloBypHW.m_flow) annotation (Line(points={{87,-278},
              {72,-278},{72,-70},{-96,-70},{-96,-245},{-76,-245}}, color={0,0,127}));
      connect(valBypBoi.port_a, senHWFlo.port_a) annotation (Line(points={{220,-252},
              {98,-252},{98,-268}}, color={238,46,47},
          thickness=0.5));
      connect(THWRet.port_b, senHWFlo.port_a)
        annotation (Line(points={{98,-236},{98,-268}}, color={238,46,47},
          thickness=0.5));
      connect(senHWFlo.port_b, boi.port_a) annotation (Line(points={{98,-288},{98,
              -320},{130,-320}}, color={238,46,47},
          thickness=0.5));
      connect(valBypBoi.port_b, resHW.port_b) annotation (Line(points={{240,-252},{
              320,-252},{320,-290},{350,-290},{350,-270}}, color={238,46,47},
          thickness=0.5));
      connect(boiTSup.y, proBoi.u2)
        annotation (Line(points={{-53,-278},{-34,-278}}, color={0,0,127}));
      connect(proBoi.y, boi.y) annotation (Line(points={{-10,-272},{12,-272},{12,
              -358},{120,-358},{120,-312},{128,-312}},
                                 color={0,0,127}));
      connect(triResHW.dpSet, dpSetGaiHW.u) annotation (Line(points={{-135,-221},{
              -128,-221},{-128,-300},{-122,-300}}, color={0,0,127}));
      connect(triResHW.TSet, boiTSup.u_s) annotation (Line(points={{-135,-231},{
              -100,-231},{-100,-278},{-76,-278}},
                                            color={0,0,127}));
      connect(HWVal.y_actual, triResHW.u) annotation (Line(points={{91,-185},{91,
              -200},{70,-200},{70,-66},{-180,-66},{-180,-226},{-158,-226}},
                                                          color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false, extent={{-200,
                -380},{350,-60}})),                                  Diagram(
            coordinateSystem(preserveAspectRatio=false, extent={{-200,-380},{350,
                -60}})));
    end PartialHotWaterside;

    partial model PartialPhysicalAirside
      "Partial model of variable air volume flow system with terminal reheat and five 
  thermal zones: this is a copy of Buildings.Examples.VAVReheat.BaseClasses.PartialOpenLoop"

      package MediumA = Buildings.Media.Air "Medium model for air";
      package MediumW = Buildings.Media.Water "Medium model for water";

      constant Integer numZon=5 "Total number of served VAV boxes";

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
        dp1_nominal=0,
        dp2_nominal=200 + 200 + 100 + 40,
        allowFlowReversal1=false,
        allowFlowReversal2=allowFlowReversal,
        T_a1_nominal=318.15,
        T_a2_nominal=281.65) "Heating coil"
        annotation (Placement(transformation(extent={{118,-36},{98,-56}})));

      Buildings.Fluid.HeatExchangers.WetCoilCounterFlow cooCoi(
        UA_nominal=-designCoolLoad/
            Buildings.Fluid.HeatExchangers.BaseClasses.lmtd(
            T_a1=26.2,
            T_b1=12.8,
            T_a2=6,
            T_b2=12),
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

      Buildings.Fluid.Sensors.VolumeFlowRate senSupFlo(redeclare package Medium =
            MediumA, m_flow_nominal=m_flow_nominal)
        "Sensor for supply fan flow rate"
        annotation (Placement(transformation(extent={{400,-50},{420,-30}})));

      Buildings.Fluid.Sensors.VolumeFlowRate senRetFlo(redeclare package Medium =
            MediumA, m_flow_nominal=m_flow_nominal)
        "Sensor for return fan flow rate"
        annotation (Placement(transformation(extent={{360,130},{340,150}})));

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
        annotation (Placement(transformation(extent={{-318,-80},{-298,-60}})));
      Buildings.Utilities.Math.Min min(nin=5) "Computes lowest room temperature"
        annotation (Placement(transformation(extent={{1200,440},{1220,460}})));
      Buildings.Utilities.Math.Average ave(nin=5)
        "Compute average of room temperatures"
        annotation (Placement(transformation(extent={{1200,410},{1220,430}})));
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
      Buildings.Fluid.Sensors.VolumeFlowRate VOut1(redeclare package Medium =
            MediumA, m_flow_nominal=m_flow_nominal) "Outside air volume flow rate"
        annotation (Placement(transformation(extent={{-72,-44},{-50,-22}})));

      FiveZone.VAVReheat.ThermalZones.VAVBranch cor(
        redeclare package MediumA = MediumA,
        redeclare package MediumW = MediumW,
        m_flow_nominal=mCor_flow_nominal,
        VRoo=VRooCor,
        allowFlowReversal=allowFlowReversal)
        "Zone for core of buildings (azimuth will be neglected)"
        annotation (Placement(transformation(extent={{570,22},{610,62}})));
      FiveZone.VAVReheat.ThermalZones.VAVBranch sou(
        redeclare package MediumA = MediumA,
        redeclare package MediumW = MediumW,
        m_flow_nominal=mSou_flow_nominal,
        VRoo=VRooSou,
        allowFlowReversal=allowFlowReversal) "South-facing thermal zone"
        annotation (Placement(transformation(extent={{750,20},{790,60}})));
      FiveZone.VAVReheat.ThermalZones.VAVBranch eas(
        redeclare package MediumA = MediumA,
        redeclare package MediumW = MediumW,
        m_flow_nominal=mEas_flow_nominal,
        VRoo=VRooEas,
        allowFlowReversal=allowFlowReversal) "East-facing thermal zone"
        annotation (Placement(transformation(extent={{930,20},{970,60}})));
      FiveZone.VAVReheat.ThermalZones.VAVBranch nor(
        redeclare package MediumA = MediumA,
        redeclare package MediumW = MediumW,
        m_flow_nominal=mNor_flow_nominal,
        VRoo=VRooNor,
        allowFlowReversal=allowFlowReversal) "North-facing thermal zone"
        annotation (Placement(transformation(extent={{1090,20},{1130,60}})));
      FiveZone.VAVReheat.ThermalZones.VAVBranch wes(
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
            Modelica.Utilities.Files.loadResource("modelica://Buildings/Resources/weatherdata/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.mos"))
        annotation (Placement(transformation(extent={{-360,170},{-340,190}})));
      Buildings.BoundaryConditions.WeatherData.Bus weaBus "Weather Data Bus"
        annotation (Placement(transformation(extent={{-330,170},{-310,190}}),
            iconTransformation(extent={{-360,170},{-340,190}})));
      FiveZone.VAVReheat.ThermalZones.Floor flo(
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
      FiveZone.VAVReheat.BaseClasses.MixingBox eco(
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
          color={170,213,255},
          thickness=0.5));
      connect(splRetRoo1.port_1, dpRetDuc.port_a) annotation (Line(
          points={{630,0},{430,0},{430,140},{400,140}},
          color={170,213,255},
          thickness=0.5));
      connect(splRetNor.port_1, splRetEas.port_2) annotation (Line(
          points={{1142,0},{1110,0},{1110,0},{1078,0},{1078,0},{1012,0}},
          color={170,213,255},
          thickness=0.5));
      connect(splRetEas.port_1, splRetSou.port_2) annotation (Line(
          points={{992,0},{952,0},{952,0},{912,0},{912,0},{832,0}},
          color={170,213,255},
          thickness=0.5));
      connect(splRetSou.port_1, splRetRoo1.port_2) annotation (Line(
          points={{812,0},{650,0}},
          color={170,213,255},
          thickness=0.5));
      connect(splSupRoo1.port_3, cor.port_a) annotation (Line(
          points={{580,-30},{580,22}},
          color={170,213,255},
          thickness=0.5));
      connect(splSupRoo1.port_2, splSupSou.port_1) annotation (Line(
          points={{590,-40},{750,-40}},
          color={170,213,255},
          thickness=0.5));
      connect(splSupSou.port_3, sou.port_a) annotation (Line(
          points={{760,-30},{760,20}},
          color={170,213,255},
          thickness=0.5));
      connect(splSupSou.port_2, splSupEas.port_1) annotation (Line(
          points={{770,-40},{930,-40}},
          color={170,213,255},
          thickness=0.5));
      connect(splSupEas.port_3, eas.port_a) annotation (Line(
          points={{940,-30},{940,20}},
          color={170,213,255},
          thickness=0.5));
      connect(splSupEas.port_2, splSupNor.port_1) annotation (Line(
          points={{950,-40},{1090,-40}},
          color={170,213,255},
          thickness=0.5));
      connect(splSupNor.port_3, nor.port_a) annotation (Line(
          points={{1100,-30},{1100,20}},
          color={170,213,255},
          thickness=0.5));
      connect(splSupNor.port_2, wes.port_a) annotation (Line(
          points={{1110,-40},{1300,-40},{1300,20}},
          color={170,213,255},
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
          points={{640,10},{640,364},{874,364},{874,472},{898,472},{898,449.533},
              {924.286,449.533}},
          color={170,213,255},
          thickness=0.5));
      connect(splRetSou.port_3, flo.portsSou[2]) annotation (Line(
          points={{822,10},{822,350},{900,350},{900,420.2},{924.286,420.2}},
          color={170,213,255},
          thickness=0.5));
      connect(splRetEas.port_3, flo.portsEas[2]) annotation (Line(
          points={{1002,10},{1002,368},{1067.2,368},{1067.2,445.867}},
          color={170,213,255},
          thickness=0.5));
      connect(splRetNor.port_3, flo.portsNor[2]) annotation (Line(
          points={{1152,10},{1152,446},{924.286,446},{924.286,478.867}},
          color={170,213,255},
          thickness=0.5));
      connect(splRetNor.port_2, flo.portsWes[2]) annotation (Line(
          points={{1162,0},{1342,0},{1342,394},{854,394},{854,449.533}},
          color={170,213,255},
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
          color={170,213,255},
          thickness=0.5));
      connect(cor.port_b, TSupCor.port_a) annotation (Line(
          points={{580,62},{580,82}},
          color={170,213,255},
          thickness=0.5));

      connect(sou.port_b, TSupSou.port_a) annotation (Line(
          points={{760,60},{760,82}},
          color={170,213,255},
          thickness=0.5));
      connect(eas.port_b, TSupEas.port_a) annotation (Line(
          points={{940,60},{940,80}},
          color={170,213,255},
          thickness=0.5));
      connect(nor.port_b, TSupNor.port_a) annotation (Line(
          points={{1100,60},{1100,84}},
          color={170,213,255},
          thickness=0.5));
      connect(wes.port_b, TSupWes.port_a) annotation (Line(
          points={{1300,60},{1300,80}},
          color={170,213,255},
          thickness=0.5));

      connect(TSupCor.port_b, VSupCor_flow.port_a) annotation (Line(
          points={{580,102},{580,120}},
          color={170,213,255},
          thickness=0.5));
      connect(TSupSou.port_b, VSupSou_flow.port_a) annotation (Line(
          points={{760,102},{760,120}},
          color={170,213,255},
          thickness=0.5));
      connect(TSupEas.port_b, VSupEas_flow.port_a) annotation (Line(
          points={{940,100},{940,100},{940,118}},
          color={170,213,255},
          thickness=0.5));
      connect(TSupNor.port_b, VSupNor_flow.port_a) annotation (Line(
          points={{1100,104},{1100,122}},
          color={170,213,255},
          thickness=0.5));
      connect(TSupWes.port_b, VSupWes_flow.port_a) annotation (Line(
          points={{1300,100},{1300,118}},
          color={170,213,255},
          thickness=0.5));
      connect(VSupCor_flow.port_b, flo.portsCor[1]) annotation (Line(
          points={{580,140},{580,372},{866,372},{866,480},{912.571,480},{
              912.571,449.533}},
          color={170,213,255},
          thickness=0.5));

      connect(VSupSou_flow.port_b, flo.portsSou[1]) annotation (Line(
          points={{760,140},{760,356},{912.571,356},{912.571,420.2}},
          color={170,213,255},
          thickness=0.5));
      connect(VSupEas_flow.port_b, flo.portsEas[1]) annotation (Line(
          points={{940,138},{940,376},{1055.49,376},{1055.49,445.867}},
          color={170,213,255},
          thickness=0.5));
      connect(VSupNor_flow.port_b, flo.portsNor[1]) annotation (Line(
          points={{1100,142},{1100,498},{912.571,498},{912.571,478.867}},
          color={170,213,255},
          thickness=0.5));
      connect(VSupWes_flow.port_b, flo.portsWes[1]) annotation (Line(
          points={{1300,138},{1300,384},{842.286,384},{842.286,449.533}},
          color={170,213,255},
          thickness=0.5));
      connect(VOut1.port_b, eco.port_Out) annotation (Line(
          points={{-50,-33},{-42,-33},{-42,-40},{-20,-40}},
          color={170,213,255},
          thickness=0.5));
      connect(eco.port_Sup, TMix.port_a) annotation (Line(
          points={{0,-40},{30,-40}},
          color={170,213,255},
          thickness=0.5));
      connect(eco.port_Exh, amb.ports[2]) annotation (Line(
          points={{-20,-52},{-96,-52},{-96,-45},{-114,-45}},
          color={170,213,255},
          thickness=0.5));
      connect(eco.port_Ret, TRet.port_b) annotation (Line(
          points={{0,-52},{10,-52},{10,140},{90,140}},
          color={170,213,255},
          thickness=0.5));
      connect(senRetFlo.port_a, dpRetDuc.port_b)
        annotation (Line(points={{360,140},{380,140}}, color={170,213,255},
          thickness=0.5));
      connect(TSup.port_b, senSupFlo.port_a)
        annotation (Line(points={{350,-40},{400,-40}}, color={170,213,255},
          thickness=0.5));
      connect(senSupFlo.port_b, splSupRoo1.port_1)
        annotation (Line(points={{420,-40},{570,-40}}, color={170,213,255},
          thickness=0.5));
      connect(dpDisSupFan.port_b, amb.ports[3]) annotation (Line(
          points={{320,10},{320,14},{-88,14},{-88,-47.9333},{-114,-47.9333}},
          color={0,0,0},
          pattern=LinePattern.Dot));
      connect(senRetFlo.port_b, TRet.port_a) annotation (Line(points={{340,140},{
              226,140},{110,140}}, color={170,213,255},
          thickness=0.5));
      connect(freStaTSetPoi.y, freSta.reference)
        annotation (Line(points={{-18,-86},{-2,-86}}, color={0,0,127}));
      connect(freSta.u, TMix.T) annotation (Line(points={{-2,-98},{-10,-98},{-10,-70},
              {20,-70},{20,-20},{40,-20},{40,-29}}, color={0,0,127}));
      connect(TMix.port_b, heaCoi.port_a2) annotation (Line(
          points={{50,-40},{98,-40}},
          color={170,213,255},
          thickness=0.5));
      connect(heaCoi.port_b2, cooCoi.port_a2) annotation (Line(
          points={{118,-40},{190,-40}},
          color={170,213,255},
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
</html>",     revisions="<html>
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
    end PartialPhysicalAirside;

    partial model PartialAirside
      "Variable air volume flow system with terminal reheat and five thermal zones"
      extends FiveZone.BaseClasses.PartialPhysicalAirside(heaCoi(dp1_nominal=
              30000));

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
        final samplePeriod=samplePeriod) "Controller for terminal unit corridor"
        annotation (Placement(transformation(extent={{530,32},{550,52}})));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVSou(
        V_flow_nominal=mSou_flow_nominal/1.2,
        AFlo=AFloSou,
        final samplePeriod=samplePeriod) "Controller for terminal unit south"
        annotation (Placement(transformation(extent={{700,30},{720,50}})));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVEas(
        V_flow_nominal=mEas_flow_nominal/1.2,
        AFlo=AFloEas,
        final samplePeriod=samplePeriod) "Controller for terminal unit east"
        annotation (Placement(transformation(extent={{880,30},{900,50}})));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVNor(
        V_flow_nominal=mNor_flow_nominal/1.2,
        AFlo=AFloNor,
        final samplePeriod=samplePeriod) "Controller for terminal unit north"
        annotation (Placement(transformation(extent={{1040,30},{1060,50}})));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.Controller conVAVWes(
        V_flow_nominal=mWes_flow_nominal/1.2,
        AFlo=AFloWes,
        final samplePeriod=samplePeriod) "Controller for terminal unit west"
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
        annotation (Placement(transformation(extent={{20,-140},{40,-120}})));
      Buildings.Controls.OBC.CDL.Continuous.Sources.Constant yFreHeaCoi(final k=0.3)
        "Flow rate signal for heating coil when freeze stat is on"
        annotation (Placement(transformation(extent={{-80,-132},{-60,-112}})));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.TerminalUnits.ModeAndSetPoints TZonSet[
        numZon](
        final TZonHeaOn=fill(THeaOn, numZon),
        final TZonHeaOff=fill(THeaOff, numZon),
        TZonCooOn=fill(TCooOn, numZon),
        final TZonCooOff=fill(TCooOff, numZon)) "Zone setpoint temperature"
        annotation (Placement(transformation(extent={{60,300},{80,320}})));
      Buildings.Controls.OBC.CDL.Routing.BooleanReplicator booRep(
        final nout=numZon)
        "Replicate boolean input"
        annotation (Placement(transformation(extent={{-120,280},{-100,300}})));
      Buildings.Controls.OBC.CDL.Routing.RealReplicator reaRep(
        final nout=numZon)
        "Replicate real input"
        annotation (Placement(transformation(extent={{-120,320},{-100,340}})));
      FiveZone.VAVReheat.Controls.ControllerOve conAHU(
        final pMaxSet=410,
        final yFanMin=yFanMin,
        final VPriSysMax_flow=VPriSysMax_flow,
        final peaSysPop=1.2*sum({0.05*AFlo[i] for i in 1:numZon})) "AHU controller"
        annotation (Placement(transformation(extent={{340,512},{420,640}})));
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

      Modelica.Blocks.Logical.And andFreSta
        annotation (Placement(transformation(extent={{-20,-140},{0,-120}})));
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
      connect(TSupNor.T, conVAVNor.TDis) annotation (Line(points={{1089,94},{1032,
              94},{1032,34},{1038,34}}, color={0,0,127}));
      connect(TSupWes.T, conVAVWes.TDis) annotation (Line(points={{1289,90},{1228,
              90},{1228,32},{1238,32}}, color={0,0,127}));
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
      connect(TSup.T, conVAVCor.TSupAHU) annotation (Line(points={{340,-29},{340,
              -20},{514,-20},{514,34},{528,34}},
                                            color={0,0,127}));
      connect(TSup.T, conVAVSou.TSupAHU) annotation (Line(points={{340,-29},{340,
              -20},{686,-20},{686,32},{698,32}},
                                            color={0,0,127}));
      connect(TSup.T, conVAVEas.TSupAHU) annotation (Line(points={{340,-29},{340,
              -20},{864,-20},{864,32},{878,32}},
                                            color={0,0,127}));
      connect(TSup.T, conVAVNor.TSupAHU) annotation (Line(points={{340,-29},{340,
              -20},{1028,-20},{1028,32},{1038,32}},
                                               color={0,0,127}));
      connect(TSup.T, conVAVWes.TSupAHU) annotation (Line(points={{340,-29},{340,
              -20},{1224,-20},{1224,30},{1238,30}},
                                               color={0,0,127}));
      connect(yOutDam.y, eco.yExh)
        annotation (Line(points={{-18,-10},{-3,-10},{-3,-34}}, color={0,0,127}));
      connect(yFreHeaCoi.y, swiFreSta.u1) annotation (Line(points={{-58,-122},{18,
              -122}},               color={0,0,127}));
      connect(TZonSet[1].yOpeMod, conVAVCor.uOpeMod) annotation (Line(points={{82,303},
              {130,303},{130,180},{420,180},{420,14},{520,14},{520,32},{528,32}},
            color={255,127,0}));
      connect(flo.TRooAir, TZonSet.TZon) annotation (Line(points={{1094.14,
              491.333},{1164,491.333},{1164,666},{50,666},{50,313},{58,313}},
                                                                     color={0,0,127}));
      connect(occSch.occupied, booRep.u) annotation (Line(points={{-297,-76},{-160,
              -76},{-160,290},{-122,290}},  color={255,0,255}));
      connect(occSch.tNexOcc, reaRep.u) annotation (Line(points={{-297,-64},{-180,
              -64},{-180,330},{-122,330}},
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
              {308,589},{308,609.778},{336,609.778}},           color={0,0,127}));
      connect(zonToSys.VSumDesPopBreZon_flow, conAHU.VSumDesPopBreZon_flow)
        annotation (Line(points={{302,586},{310,586},{310,604.444},{336,604.444}},
            color={0,0,127}));
      connect(zonToSys.VSumDesAreBreZon_flow, conAHU.VSumDesAreBreZon_flow)
        annotation (Line(points={{302,583},{312,583},{312,599.111},{336,599.111}},
            color={0,0,127}));
      connect(zonToSys.yDesSysVenEff, conAHU.uDesSysVenEff) annotation (Line(points={{302,580},
              {314,580},{314,593.778},{336,593.778}},           color={0,0,127}));
      connect(zonToSys.VSumUncOutAir_flow, conAHU.VSumUncOutAir_flow) annotation (
          Line(points={{302,577},{316,577},{316,588.444},{336,588.444}}, color={0,0,
              127}));
      connect(zonToSys.VSumSysPriAir_flow, conAHU.VSumSysPriAir_flow) annotation (
          Line(points={{302,571},{318,571},{318,583.111},{336,583.111}}, color={0,0,
              127}));
      connect(zonToSys.uOutAirFra_max, conAHU.uOutAirFra_max) annotation (Line(
            points={{302,574},{320,574},{320,577.778},{336,577.778}}, color={0,0,127}));
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
            points={{424,586.667},{440,586.667},{440,468},{270,468},{270,582},{
              278,582}},
            color={0,0,127}));
      connect(conAHU.VDesUncOutAir_flow, reaRep1.u) annotation (Line(points={{424,
              597.333},{440,597.333},{440,590},{458,590}},
                                                  color={0,0,127}));
      connect(reaRep1.y, zonOutAirSet.VUncOut_flow_nominal) annotation (Line(points={{482,590},
              {490,590},{490,464},{210,464},{210,581},{218,581}},          color={0,
              0,127}));
      connect(conAHU.yReqOutAir, booRep1.u) annotation (Line(points={{424,
              565.333},{444,565.333},{444,560},{458,560}},
                                                 color={255,0,255}));
      connect(booRep1.y, zonOutAirSet.uReqOutAir) annotation (Line(points={{482,560},
              {496,560},{496,460},{206,460},{206,593},{218,593}}, color={255,0,255}));
      connect(flo.TRooAir, zonOutAirSet.TZon) annotation (Line(points={{1094.14,
              491.333},{1166,491.333},{1166,672},{210,672},{210,590},{218,590}},
                                                                        color={0,0,127}));
      connect(TDis.y, zonOutAirSet.TDis) annotation (Line(points={{241,280},{252,280},
              {252,340},{200,340},{200,587},{218,587}}, color={0,0,127}));
      connect(VDis_flow.y, zonOutAirSet.VDis_flow) annotation (Line(points={{241,240},
              {260,240},{260,346},{194,346},{194,584},{218,584}}, color={0,0,127}));
      connect(TZonSet[1].yOpeMod, conAHU.uOpeMod) annotation (Line(points={{82,303},
              {140,303},{140,531.556},{336,531.556}}, color={255,127,0}));
      connect(TZonResReq.y, conAHU.uZonTemResReq) annotation (Line(points={{322,370},
              {330,370},{330,526.222},{336,526.222}}, color={255,127,0}));
      connect(PZonResReq.y, conAHU.uZonPreResReq) annotation (Line(points={{322,340},
              {326,340},{326,520.889},{336,520.889}}, color={255,127,0}));
      connect(TZonSet[1].TZonHeaSet, conAHU.TZonHeaSet) annotation (Line(points={{82,310},
              {110,310},{110,636.444},{336,636.444}},      color={0,0,127}));
      connect(TZonSet[1].TZonCooSet, conAHU.TZonCooSet) annotation (Line(points={{82,317},
              {120,317},{120,631.111},{336,631.111}},      color={0,0,127}));
      connect(TOut.y, conAHU.TOut) annotation (Line(points={{-279,180},{-260,
              180},{-260,625.778},{336,625.778}},
                                       color={0,0,127}));
      connect(dpDisSupFan.p_rel, conAHU.ducStaPre) annotation (Line(points={{311,0},
              {160,0},{160,620.444},{336,620.444}}, color={0,0,127}));
      connect(TSup.T, conAHU.TSup) annotation (Line(points={{340,-29},{340,-20},
              {152,-20},{152,567.111},{336,567.111}},
                                                 color={0,0,127}));
      connect(TRet.T, conAHU.TOutCut) annotation (Line(points={{100,151},{100,
              561.778},{336,561.778}},
                              color={0,0,127}));
      connect(VOut1.V_flow, conAHU.VOut_flow) annotation (Line(points={{-61,
              -20.9},{-61,545.778},{336,545.778}},
                                           color={0,0,127}));
      connect(TMix.T, conAHU.TMix) annotation (Line(points={{40,-29},{40,
              538.667},{336,538.667}},
                         color={0,0,127}));
      connect(conAHU.yOutDamPos, eco.yOut) annotation (Line(points={{424,
              522.667},{448,522.667},{448,36},{-10,36},{-10,-34}},
                                                     color={0,0,127}));
      connect(conAHU.yRetDamPos, eco.yRet) annotation (Line(points={{424,
              533.333},{442,533.333},{442,40},{-16.8,40},{-16.8,-34}},
                                                         color={0,0,127}));
      connect(conAHU.yHea, swiFreSta.u3) annotation (Line(points={{424,554.667},
              {452,554.667},{452,32},{22,32},{22,-108},{10,-108},{10,-138},{18,
              -138}},                                             color={0,0,127}));
      connect(conAHU.ySupFanSpe, fanSup.y) annotation (Line(points={{424,
              618.667},{432,618.667},{432,-14},{310,-14},{310,-28}},
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
      connect(andFreSta.y, swiFreSta.u2)
        annotation (Line(points={{1,-130},{18,-130}},  color={255,0,255}));
      connect(freSta.y, andFreSta.u1) annotation (Line(points={{22,-92},{28,-92},{
              28,-112},{-40,-112},{-40,-130},{-22,-130}},
                                                     color={255,0,255}));
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
</html>",     revisions="<html>
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
        experiment(StopTime=172800, Tolerance=1e-06),
        Icon(coordinateSystem(extent={{-100,-100},{100,100}})));
    end PartialAirside;

    partial model EnergyMeter "System example for fault injection"

     Modelica.Blocks.Sources.RealExpression eleSupFan "Pow of fan"
        annotation (Placement(transformation(extent={{1224,672},{1244,692}})));
      Modelica.Blocks.Sources.RealExpression eleChi
        "Power of chiller"
        annotation (Placement(transformation(extent={{1224,652},{1244,672}})));
      Modelica.Blocks.Sources.RealExpression eleCHWP
        "Power of chilled water pump"
        annotation (Placement(transformation(extent={{1224,632},{1244,652}})));
      Modelica.Blocks.Sources.RealExpression eleCWP "Power of CWP"
        annotation (Placement(transformation(extent={{1224,612},{1244,632}})));
      Modelica.Blocks.Sources.RealExpression eleCT
        "Power of cooling tower"
        annotation (Placement(transformation(extent={{1224,592},{1244,612}})));
      Modelica.Blocks.Sources.RealExpression eleHWP
        "Power of hot water pump"
        annotation (Placement(transformation(extent={{1224,572},{1244,592}})));
      Modelica.Blocks.Sources.RealExpression eleCoiVAV
        "Power of VAV terminal reheat coil"
        annotation (Placement(transformation(extent={{1224,694},{1244,714}})));
      Modelica.Blocks.Sources.RealExpression gasBoi
        "Gas consumption of gas boiler"
        annotation (Placement(transformation(extent={{1224,544},{1244,564}})));
      Modelica.Blocks.Math.MultiSum eleTot(nu=7) "Electricity in total"
        annotation (Placement(transformation(extent={{1288,698},{1300,710}})));

    equation
      connect(eleCoiVAV.y, eleTot.u[1]) annotation (Line(points={{1245,704},{
              1266,704},{1266,707.6},{1288,707.6}},
                                               color={0,0,127}));
      connect(eleSupFan.y, eleTot.u[2]) annotation (Line(points={{1245,682},{
              1266.5,682},{1266.5,706.4},{1288,706.4}},
                                                 color={0,0,127}));
      connect(eleChi.y, eleTot.u[3]) annotation (Line(points={{1245,662},{1268,
              662},{1268,705.2},{1288,705.2}},
                                          color={0,0,127}));
      connect(eleCHWP.y, eleTot.u[4]) annotation (Line(points={{1245,642},{1270,
              642},{1270,704},{1288,704}},
                                      color={0,0,127}));
      connect(eleCWP.y, eleTot.u[5]) annotation (Line(points={{1245,622},{1272,
              622},{1272,702.8},{1288,702.8}},
                                          color={0,0,127}));
      connect(eleCT.y, eleTot.u[6]) annotation (Line(points={{1245,602},{1274,
              602},{1274,701.6},{1288,701.6}},
                                          color={0,0,127}));
      connect(eleHWP.y, eleTot.u[7]) annotation (Line(points={{1245,582},{1276,
              582},{1276,700.4},{1288,700.4}},
                                          color={0,0,127}));
      annotation (Diagram(coordinateSystem(extent={{-100,-100},{1580,700}})), Icon(
            coordinateSystem(extent={{-100,-100},{1580,700}})));
    end EnergyMeter;
  end BaseClasses;

  package Controls
    extends Modelica.Icons.Package;

    model ChillerStage "Chiller staging control logic"
      extends Modelica.Blocks.Icons.Block;

      parameter Modelica.SIunits.Time tWai "Waiting time";

      Modelica.Blocks.Interfaces.IntegerInput cooMod
        "Cooling mode signal, integer value of
    Buildings.Applications.DataCenters.Types.CoolingMode"
        annotation (Placement(transformation(extent={{-140,40},{-100,80}})));
      Modelica.Blocks.Interfaces.RealOutput y
        "On/off signal for the chillers - 0: off; 1: on"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));

      Modelica.StateGraph.Transition con1(
        enableTimer=true,
        waitTime=tWai,
        condition=cooMod > Integer(FiveZone.Types.CoolingModes.FreeCooling)
             and cooMod < Integer(FiveZone.Types.CoolingModes.Off))
        "Fire condition 1: free cooling to partially mechanical cooling"
        annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={-50,42})));
      Modelica.StateGraph.StepWithSignal oneOn(nIn=2, nOut=2)
        "One chiller is commanded on"
        annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-50,10})));
      Modelica.StateGraph.InitialStep off(nIn=1) "Free cooling mode"
        annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-50,70})));
      Modelica.StateGraph.Transition con4(
        enableTimer=true,
        waitTime=tWai,
        condition=cooMod == Integer(FiveZone.Types.CoolingModes.FreeCooling)
             or cooMod == Integer(FiveZone.Types.CoolingModes.Off))
        "Fire condition 4: partially mechanical cooling to free cooling"
        annotation (Placement(transformation(
            extent={{10,-10},{-10,10}},
            rotation=-90,
            origin={-20,52})));
      inner Modelica.StateGraph.StateGraphRoot stateGraphRoot
        annotation (Placement(transformation(extent={{40,60},{60,80}})));

      Buildings.Controls.OBC.CDL.Conversions.BooleanToReal    booToRea
        annotation (Placement(transformation(extent={{20,-10},{40,10}})));
    equation
      connect(off.outPort[1], con1.inPort)
        annotation (Line(
          points={{-50,59.5},{-50,46}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con1.outPort, oneOn.inPort[1])
        annotation (Line(
          points={{-50,40.5},{-50,26},{-50.5,26},{-50.5,21}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con4.outPort, off.inPort[1])
        annotation (Line(
          points={{-20,53.5},{-20,90},{-50,90},{-50,81}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con4.inPort, oneOn.outPort[2])
        annotation (Line(
          points={{-20,48},{-20,-10},{-49.75,-10},{-49.75,-0.5}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(oneOn.active, booToRea.u) annotation (Line(points={{-39,10},{-12,10},{
              -12,0},{18,0}}, color={255,0,255}));
      connect(booToRea.y, y)
        annotation (Line(points={{42,0},{70,0},{70,0},{110,0}}, color={0,0,127}));
      annotation (Documentation(info="<html>
<p>
This is a chiller staging control that works as follows:
</p>
<ul>
<li>
The chillers are all off when cooling mode is Free Cooling.
</li>
<li>
One chiller is commanded on when cooling mode is not Free Cooling.
</li>
<li>
Two chillers are commanded on when cooling mode is not Free Cooling
and the cooling load addressed by each chiller is larger than
a critical value.
</li>
</ul>
</html>",     revisions="<html>
<ul>
<li>
September 11, 2017, by Michael Wetter:<br/>
Revised switch that selects the operation mode for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/921\">issue 921</a>
</li>
<li>
July 30, 2017, by Yangyang Fu:<br/>
First implementation.
</li>
</ul>
</html>"));
    end ChillerStage;

    model ChillerPlantEnableDisable
      "Chilled water plant enable disable control sequence"
      extends Modelica.Blocks.Icons.Block;

      parameter Integer numIgn=0 "Number of ignored plant requests";

      parameter Real yFanSpeMin(min=0.1, max=1, unit="1") = 0.15
        "Lowest allowed fan speed if fan is on";

      parameter Modelica.SIunits.Time shoCycTim=15*60 "Time duration to avoid short cycling of equipment";

      parameter Modelica.SIunits.Time plaReqTim=3*60 "Time duration of plant requests";

      parameter Modelica.SIunits.Time tWai=60 "Waiting time";

      parameter Modelica.SIunits.Temperature TOutPla = 13+273.15
        "The outdoor air lockout temperature below/over which the chiller/boiler plant is prevented from operating.
    It is typically 13°C for chiller plants serving systems with airside economizers. 
    For boiler plant, it is normally 18°C";

      Modelica.StateGraph.Transition con1(
        condition=yPlaReq > numIgn and TOut > TOutPla and ySupFan and offTim.y >=
            shoCycTim,
        enableTimer=true,
        waitTime=tWai)
        "Fire condition 1: plant off to on"
        annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={-50,32})));
      Modelica.StateGraph.StepWithSignal On(nIn=1, nOut=1) "Plant is commanded on"
        annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-50,0})));
      Modelica.StateGraph.InitialStepWithSignal
                                      off(nIn=1) "Plant is off"
        annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-50,60})));
      Modelica.StateGraph.Transition con2(
        condition=(lesEquReq.y >= plaReqTim and onTim.y >= shoCycTim and lesEquSpe.y
             >= plaReqTim) or ((TOut < TOutPla - 1 or not ySupFan) and onTim.y >=
            shoCycTim),
        enableTimer=true,
        waitTime=0) "Fire condition 2: plant on to off" annotation (Placement(
            transformation(
            extent={{10,-10},{-10,10}},
            rotation=-90,
            origin={-18,34})));
      inner Modelica.StateGraph.StateGraphRoot stateGraphRoot
        annotation (Placement(transformation(extent={{40,60},{60,80}})));

      Buildings.Controls.OBC.CDL.Interfaces.RealInput TOut(final unit="K", final
          quantity="ThermodynamicTemperature")     "Outdoor air temperature"
        annotation (Placement(transformation(extent={{-140,26},{-100,66}}),
            iconTransformation(extent={{-140,26},{-100,66}})));
      Buildings.Controls.OBC.CDL.Interfaces.BooleanInput ySupFan
        "Supply fan on status"
        annotation (Placement(transformation(extent={{-140,-20},{-100,20}}),
            iconTransformation(extent={{-140,-20},{-100,20}})));
      Buildings.Controls.OBC.CDL.Interfaces.IntegerInput yPlaReq "Plant request"
        annotation (Placement(transformation(extent={{-140,50},{-100,90}}),
          iconTransformation(extent={{-140,-60},{-100,-20}})));
      Modelica.Blocks.Logical.Timer offTim
        "Timer for the state where equipment is off"
        annotation (Placement(transformation(extent={{-8,50},{12,70}})));
      Modelica.Blocks.Logical.Timer onTim
        "Timer for the state where equipment is on"
        annotation (Placement(transformation(extent={{-10,-40},{10,-20}})));
      FiveZone.Controls.BaseClasses.TimeLessEqual lesEquReq(threshold=numIgn)
        annotation (Placement(transformation(extent={{-90,60},{-70,80}})));

      Modelica.Blocks.Interfaces.BooleanOutput yPla
        "On/off signal for the plant - 0: off; 1: on"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      Modelica.Blocks.Interfaces.RealInput yFanSpe(unit="1")
        "Constant normalized rotational speed" annotation (Placement(transformation(
            extent={{-20,-20},{20,20}},
            rotation=0,
            origin={-120,-40}), iconTransformation(
            extent={{-10,-10},{10,10}},
            rotation=0,
            origin={-110,-70})));
      BaseClasses.TimeLessEqualRea lesEquSpe(threshold=yFanSpeMin)
        annotation (Placement(transformation(extent={{-90,-50},{-70,-30}})));
    equation
      connect(off.outPort[1], con1.inPort)
        annotation (Line(
          points={{-50,49.5},{-50,36}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con1.outPort, On.inPort[1]) annotation (Line(
          points={{-50,30.5},{-50,16},{-50,16},{-50,11}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con2.outPort, off.inPort[1])
        annotation (Line(
          points={{-18,35.5},{-18,80},{-50,80},{-50,71}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(off.active, offTim.u)
        annotation (Line(points={{-39,60},{-10,60}}, color={255,0,255}));
      connect(On.active, onTim.u) annotation (Line(points={{-39,0},{-30,0},{-30,-30},
              {-12,-30}}, color={255,0,255}));
      connect(yPlaReq, lesEquReq.u1)
        annotation (Line(points={{-120,70},{-92,70}}, color={255,127,0}));
      connect(On.active, yPla) annotation (Line(points={{-39,-1.9984e-15},{8,
              -1.9984e-15},{8,0},{110,0}},   color={255,0,255}));
      connect(On.outPort[1], con2.inPort) annotation (Line(
          points={{-50,-10.5},{-50,-20},{-18,-20},{-18,30}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(yFanSpe, lesEquSpe.u1)
        annotation (Line(points={{-120,-40},{-92,-40}}, color={0,0,127}));
      annotation (Documentation(info="<html>
<p>This is a chilled plant enable disable control that works as follows: </p>
<p>Enable the plant in the lowest stage when the plant has been disabled for at least 15 minutes and: </p>
<ol>
<li>Number of Chiller Plant Requests &gt; I (I = Ignores shall default to 0, adjustable), and </li>
<li>OAT&gt;CH-LOT, and </li>
<li>The chiller plant enable schedule is active. </li>
</ol>
<p>Disable the plant when it has been enabled for at least 15 minutes and: </p>
<ol>
<li>Number of Chiller Plant Requests <span style=\"font-family: TimesNewRomanPSMT;\">&le; </span>I for 3 minutes, or </li>
<li>OAT&lt;CH-LOT<span style=\"font-family: TimesNewRomanPSMT;\">-</span>1&deg;F, or </li>
<li>The chiller plant enable schedule is inactive. </li>
</ol>
</html>",     revisions="<html>
<ul>
<li>Aug 30, 2020, by Xing Lu:<br>First implementation. </li>
</ul>
</html>"));
    end ChillerPlantEnableDisable;

    model CoolingMode
      "Mode controller for integrated waterside economizer and chiller"
      extends Modelica.Blocks.Icons.Block;

      parameter Modelica.SIunits.Time tWai "Waiting time";
      parameter Modelica.SIunits.TemperatureDifference deaBan1
        "Dead band width 1 for switching chiller on ";
      parameter Modelica.SIunits.TemperatureDifference deaBan2
        "Dead band width 2 for switching waterside economizer off";
      parameter Modelica.SIunits.TemperatureDifference deaBan3
        "Dead band width 3 for switching waterside economizer on ";
      parameter Modelica.SIunits.TemperatureDifference deaBan4
        "Dead band width 4 for switching chiller off";

      Modelica.Blocks.Interfaces.RealInput TCHWRetWSE(
        final quantity="ThermodynamicTemperature",
        final unit="K",
        displayUnit="degC")
        "Temperature of entering chilled water that flows to waterside economizer "
        annotation (Placement(transformation(extent={{-140,-100},{-100,-60}})));
      Modelica.Blocks.Interfaces.RealInput TCHWSupWSE(
        final quantity="ThermodynamicTemperature",
        final unit="K",
        displayUnit="degC")
        "Temperature of leaving chilled water that flows out from waterside economizer"
        annotation (Placement(transformation(extent={{-140,-70},{-100,-30}})));
      Modelica.Blocks.Interfaces.RealInput TCHWSupSet(
        final quantity="ThermodynamicTemperature",
        final unit="K",
        displayUnit="degC") "Supply chilled water temperature setpoint "
        annotation (Placement(transformation(extent={{-140,22},{-100,62}}),
            iconTransformation(extent={{-140,22},{-100,62}})));
      Modelica.Blocks.Interfaces.RealInput TApp(
        final quantity="TemperatureDifference",
        final unit="K",
        displayUnit="degC") "Approach temperature in the cooling tower"
        annotation (Placement(transformation(extent={{-140,-40},{-100,0}})));
      Modelica.Blocks.Interfaces.IntegerOutput y
        "Cooling mode signal, integer value of Buildings.Applications.DataCenters.Types.CoolingMode"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));

      Modelica.StateGraph.Transition con4(
        enableTimer=true,
        waitTime=tWai,
        condition=TCHWSupWSE > TCHWSupSet + deaBan1 and yPla)
        "Fire condition 4: free cooling to partially mechanical cooling"
        annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={-10,28})));
      Modelica.StateGraph.StepWithSignal parMecCoo(nIn=2, nOut=3)
        "Partial mechanical cooling mode"
        annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-10,-2})));
      Modelica.StateGraph.StepWithSignal        freCoo(nIn=1, nOut=2)
        "Free cooling mode"
        annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-10,58})));
      Modelica.StateGraph.StepWithSignal fulMecCoo(nIn=2,
                                                   nOut=2)
                                                   "Fully mechanical cooling mode"
        annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-10,-44})));
      Modelica.StateGraph.Transition con5(
        enableTimer=true,
        waitTime=tWai,
        condition=TCHWRetWSE < TCHWSupWSE + deaBan2 and yPla)
        "Fire condition 5: partially mechanical cooling to fully mechanical cooling"
        annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={-10,-24})));
      Modelica.StateGraph.Transition con2(
        enableTimer=true,
        waitTime=tWai,
        condition=TCHWRetWSE > TWetBul + TApp + deaBan3)
        "Fire condition 2: fully mechanical cooling to partially mechanical cooling"
        annotation (Placement(transformation(
            extent={{10,-10},{-10,10}},
            rotation=-90,
            origin={30,-20})));
      Modelica.StateGraph.Transition con3(
        enableTimer=true,
        waitTime=tWai,
        condition=TCHWSupWSE <= TCHWSupSet + deaBan4)
        "Fire condition 3: partially mechanical cooling to free cooling"
        annotation (Placement(transformation(
            extent={{10,-10},{-10,10}},
            rotation=-90,
            origin={20,34})));
      inner Modelica.StateGraph.StateGraphRoot stateGraphRoot
        annotation (Placement(transformation(extent={{60,60},{80,80}})));
      Modelica.Blocks.Interfaces.RealInput TWetBul(
        final quantity="ThermodynamicTemperature",
        final unit="K",
        displayUnit="degC")
        "Wet bulb temperature of outdoor air"
        annotation (Placement(transformation(extent={{-140,-10},{-100,30}})));

      Modelica.Blocks.MathInteger.MultiSwitch swi(
        y_default=0,
        expr={Integer(FiveZone.Types.CoolingModes.FreeCooling),
            Integer(FiveZone.Types.CoolingModes.PartialMechanical),
            Integer(FiveZone.Types.CoolingModes.FullMechanical),
            Integer(FiveZone.Types.CoolingModes.Off)},
        nu=4)
        "Switch boolean signals to real signal"
        annotation (Placement(transformation(extent={{68,-6},{92,6}})));

      Modelica.Blocks.Interfaces.BooleanInput            yPla "Plant on/off signal"
        annotation (Placement(transformation(extent={{-140,48},{-100,88}}),
            iconTransformation(extent={{-140,48},{-100,88}})));
      Modelica.StateGraph.InitialStepWithSignal off(nIn=3) "Off" annotation (
          Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-10,-80})));
      Modelica.StateGraph.Transition con8(
        enableTimer=true,
        waitTime=0,
        condition=not yPla) "Fire condition 8: fully mechanical cooling to off"
        annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={80,-60})));
      Modelica.StateGraph.Transition con7(
        enableTimer=true,
        waitTime=0,
        condition=not yPla) "Fire condition 7: partially mechanical cooling to off"
        annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={70,-34})));
      Modelica.StateGraph.Transition con1(
        enableTimer=true,
        waitTime=0,
        condition=yPla) "Fire condition 1: off to free cooling"
        annotation (Placement(transformation(
            extent={{10,-10},{-10,10}},
            rotation=-90,
            origin={40,-80})));
      Modelica.StateGraph.Transition con6(
        enableTimer=true,
        waitTime=0,
        condition=not yPla) "Fire condition 6: free cooling to off"
        annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={60,20})));
    equation
      connect(freCoo.outPort[1], con4.inPort) annotation (Line(
          points={{-10.25,47.5},{-10.25,32},{-10,32}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con4.outPort, parMecCoo.inPort[1]) annotation (Line(
          points={{-10,26.5},{-10,18},{-10.5,18},{-10.5,9}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con5.inPort, parMecCoo.outPort[1])
        annotation (Line(
          points={{-10,-20},{-10.3333,-20},{-10.3333,-12.5}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con5.outPort, fulMecCoo.inPort[1])
        annotation (Line(
          points={{-10,-25.5},{-10,-30},{-10,-33},{-10.5,-33}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(fulMecCoo.outPort[1],con2. inPort)
        annotation (Line(
          points={{-10.25,-54.5},{-10.25,-58},{30,-58},{30,-24}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con2.outPort, parMecCoo.inPort[2])
        annotation (Line(
          points={{30,-18.5},{30,16},{-9.5,16},{-9.5,9}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con3.inPort, parMecCoo.outPort[2]) annotation (Line(
          points={{20,30},{20,-16},{-10,-16},{-10,-12.5}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(swi.y, y)
        annotation (Line(points={{92.6,0},{110,0}}, color={255,127,0}));
      connect(parMecCoo.outPort[3],con7. inPort) annotation (Line(
          points={{-9.66667,-12.5},{-9.66667,-14},{70,-14},{70,-30}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con7.outPort, off.inPort[2]) annotation (Line(
          points={{70,-35.5},{70,-64},{-10,-64},{-10,-69}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con8.outPort, off.inPort[3]) annotation (Line(
          points={{80,-61.5},{80,-66},{-10,-66},{-10,-69},{-9.33333,-69}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(freCoo.outPort[2], con6.inPort) annotation (Line(
          points={{-9.75,47.5},{-9.75,42},{60,42},{60,24}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con6.outPort, off.inPort[1]) annotation (Line(
          points={{60,18.5},{60,-62},{-10.6667,-62},{-10.6667,-69}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(fulMecCoo.outPort[2], con8.inPort) annotation (Line(
          points={{-9.75,-54.5},{-9.75,-56},{80,-56}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(off.outPort[1], con1.inPort) annotation (Line(
          points={{-10,-90.5},{-10,-96},{40,-96},{40,-84}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con1.outPort, fulMecCoo.inPort[2]) annotation (Line(
          points={{40,-78.5},{40,-28},{-9.5,-28},{-9.5,-33}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(freCoo.active, swi.u[1]) annotation (Line(points={{1,58},{42,58},{42,
              1.35},{68,1.35}}, color={255,0,255}));
      connect(parMecCoo.active, swi.u[2]) annotation (Line(points={{1,-2},{46,-2},{
              46,0.45},{68,0.45}}, color={255,0,255}));
      connect(fulMecCoo.active, swi.u[3]) annotation (Line(points={{1,-44},{48,-44},
              {48,-0.45},{68,-0.45}}, color={255,0,255}));
      connect(off.active, swi.u[4]) annotation (Line(points={{1,-80},{20,-80},{20,
              -46},{50,-46},{50,-1.2},{68,-1.2},{68,-1.35}}, color={255,0,255}));
      connect(con3.outPort, freCoo.inPort[1]) annotation (Line(
          points={{20,35.5},{20,76},{-10,76},{-10,69}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      annotation (    Documentation(info="<html>
<p>Controller that outputs if the chilled water system is in off mode,  Free Cooling (FC) mode, Partially Mechanical Cooling (PMC) mode or Fully Mechanical Cooling (FMC) mode. </p>
<p>The waterside economizer is enabled when </p>
<ol>
<li>The waterside economizer has been disabled for at least 20 minutes, and </li>
<li><i>T<sub>CHWR</sub> &gt; T<sub>WetBul</sub> + T<sub>TowApp</sub> + deaBan1 </i></li>
</ol>
<p>The waterside economizer is disabled when </p>
<ol>
<li>The waterside economizer has been enabled for at least 20 minutes, and </li>
<li><i>T<sub>WSE_CHWST</sub> &gt; T<sub>WSE_CHWRT</sub> - deaBan2 </i></li>
</ol>
<p>The chiller is enabled when </p>
<ol>
<li>The chiller has been disabled for at leat 20 minutes, and </li>
<li><i>T<sub>WSE_CHWST</sub> &gt; T<sub>CHWSTSet</sub> + deaBan3 </i></li>
</ol>
<p>The chiller is disabled when </p>
<ol>
<li>The chiller has been enabled for at leat 20 minutes, and </li>
<li><i>T<sub>WSE_CHWST</sub> &le; T<sub>CHWSTSet</sub> + deaBan4 </i></li>
</ol>
<p>where <i>T<sub>WSE_CHWST</i></sub> is the chilled water supply temperature for the WSE, <i>T<sub>WetBul</i></sub> is the wet bulb temperature, <i>T<sub>TowApp</i></sub> is the cooling tower approach, <i>T<sub>WSE_CHWRT</i></sub> is the chilled water return temperature for the WSE, and <i>T<sub>CHWSTSet</i></sub> is the chilled water supply temperature setpoint for the system. <i>deaBan 1-4</i> are deadbands for each switching point. </p>
<h4>References</h4>
<ul>
<li>Stein, Jeff. Waterside Economizing in Data Centers: Design and Control Considerations. ASHRAE Transactions 115.2 (2009). </li>
</ul>
</html>",            revisions="<html>
<ul>
<li>
July 24, 2017, by Yangyang Fu:<br/>
First implementation.
</li>
</ul>
</html>"),
        Diagram(coordinateSystem(extent={{-100,-100},{100,80}})),
        Icon(coordinateSystem(extent={{-100,-100},{100,80}})));
    end CoolingMode;

    model ConstantSpeedPumpStage "Staging control for constant speed pumps"
      extends Modelica.Blocks.Icons.Block;

      parameter Modelica.SIunits.Time tWai "Waiting time";

      Modelica.Blocks.Interfaces.IntegerInput cooMod
        "Cooling mode - 0:off,  1: free cooling mode; 2: partially mechanical cooling; 3: fully mechanical cooling"
        annotation (Placement(transformation(extent={{-140,30},{-100,70}})));
      Modelica.Blocks.Interfaces.IntegerInput numOnChi
        "The number of running chillers"
        annotation (Placement(transformation(extent={{-140,-70},{-100,-30}})));
      Modelica.Blocks.Interfaces.RealOutput y[2] "On/off signal - 0: off; 1: on"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));

      Modelica.StateGraph.Transition con1(
        enableTimer=true,
        waitTime=tWai,
        condition=cooMod == Integer(FiveZone.Types.CoolingModes.FreeCooling)
             or cooMod == Integer(FiveZone.Types.CoolingModes.PartialMechanical)
             or cooMod == Integer(FiveZone.Types.CoolingModes.FullMechanical))
        "Fire condition 1: free cooling to partially mechanical cooling"
        annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={-40,40})));
      Modelica.StateGraph.StepWithSignal oneOn(nIn=2, nOut=2)
        "One chiller is commanded on" annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-40,10})));
      Modelica.StateGraph.InitialStep off(nIn=1) "Free cooling mode"
        annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-40,70})));
      Modelica.StateGraph.StepWithSignal twoOn "Two chillers are commanded on"
        annotation (Placement(transformation(
            extent={{-10,10},{10,-10}},
            rotation=-90,
            origin={-40,-80})));
      Modelica.StateGraph.Transition con2(
        enableTimer=true,
        waitTime=tWai,
        condition=cooMod == Integer(FiveZone.Types.CoolingModes.FreeCooling)
             or cooMod == Integer(FiveZone.Types.CoolingModes.PartialMechanical)
             or (cooMod == Integer(FiveZone.Types.CoolingModes.FullMechanical)
             and numOnChi > 1))
        "Fire condition 2: partially mechanical cooling to fully mechanical cooling"
        annotation (Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=-90,
            origin={-40,-40})));
      Modelica.StateGraph.Transition con3(
        enableTimer=true,
        waitTime=tWai,
        condition=cooMod == Integer(FiveZone.Types.CoolingModes.FullMechanical)
        and numOnChi < 2)
        "Fire condition 3: fully mechanical cooling to partially mechanical cooling"
        annotation (Placement(transformation(
            extent={{10,-10},{-10,10}},
            rotation=-90,
            origin={-10,-40})));
      Modelica.StateGraph.Transition con4(
        enableTimer=true,
        waitTime=tWai,
        condition=cooMod == Integer(FiveZone.Types.CoolingModes.Off))
        "Fire condition 4: partially mechanical cooling to free cooling"
        annotation (Placement(transformation(
            extent={{10,-10},{-10,10}},
            rotation=-90,
            origin={-8,70})));
      inner Modelica.StateGraph.StateGraphRoot stateGraphRoot
        annotation (Placement(transformation(extent={{60,60},{80,80}})));
      Modelica.Blocks.Tables.CombiTable1Ds combiTable1Ds(table=[0,0,0; 1,1,0; 2,1,1])
        annotation (Placement(transformation(extent={{70,-10},{90,10}})));

      Buildings.Controls.OBC.CDL.Conversions.BooleanToInteger booToInt(
        final integerTrue=1,
        final integerFalse=0)
        annotation (Placement(transformation(extent={{20,-50},{40,-30}})));
      Buildings.Controls.OBC.CDL.Conversions.BooleanToInteger booToInt1(
        final integerFalse=0, final integerTrue=2)
        annotation (Placement(transformation(extent={{20,-90},{40,-70}})));
      Buildings.Controls.OBC.CDL.Integers.Add addInt
        annotation (Placement(transformation(extent={{60,-70},{80,-50}})));
      Buildings.Controls.OBC.CDL.Conversions.IntegerToReal intToRea
        annotation (Placement(transformation(extent={{40,-10},{60,10}})));

    equation
      connect(off.outPort[1], con1.inPort)
        annotation (Line(
          points={{-40,59.5},{-40,44}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con1.outPort, oneOn.inPort[1])
        annotation (Line(
          points={{-40,38.5},{-40,26},{-40.5,26},{-40.5,21}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con2.inPort, oneOn.outPort[1])
        annotation (Line(
          points={{-40,-36},{-40,-10},{-40.25,-10},{-40.25,-0.5}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con2.outPort, twoOn.inPort[1])
        annotation (Line(
          points={{-40,-41.5},{-40,-69}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(twoOn.outPort[1], con3.inPort)
        annotation (Line(
          points={{-40,-90.5},{-40,-98},{-10,-98},{-10,-44}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con4.outPort, off.inPort[1])
        annotation (Line(
          points={{-8,71.5},{-8,94},{-40,94},{-40,81}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con3.outPort, oneOn.inPort[2])
        annotation (Line(
          points={{-10,-38.5},{-10,26},{-39.5,26},{-39.5,21}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(con4.inPort, oneOn.outPort[2])
        annotation (Line(
          points={{-8,66},{-8,-10},{-39.75,-10},{-39.75,-0.5}},
          color={0,0,0},
          pattern=LinePattern.Dash));
      connect(combiTable1Ds.y, y)
        annotation (Line(points={{91,0},{91,0},{110,0}},
                                                  color={0,0,127}));
      connect(oneOn.active, booToInt.u) annotation (Line(points={{-29,10},{12,10},{
              12,-40},{18,-40}},         color={255,0,255}));
      connect(twoOn.active, booToInt1.u)
        annotation (Line(points={{-29,-80},{18,-80}},          color={255,0,255}));
      connect(booToInt.y, addInt.u1) annotation (Line(points={{42,-40},{48,-40},{48,
              -54},{58,-54}}, color={255,127,0}));
      connect(booToInt1.y, addInt.u2) annotation (Line(points={{42,-80},{48,-80},{
              48,-66},{58,-66}}, color={255,127,0}));
      connect(intToRea.y, combiTable1Ds.u)
        annotation (Line(points={{62,0},{68,0}}, color={0,0,127}));
      connect(addInt.y, intToRea.u) annotation (Line(points={{82,-60},{88,-60},{88,
              -20},{30,-20},{30,0},{38,0}}, color={255,127,0}));
      annotation (                   Documentation(info="<html>
<p>
This model describes a simple staging control for two constant-speed pumps in
a chilled water plant with two chillers and a waterside economizer (WSE). The staging sequence
is shown as below.
</p>
<ul>
<li>
When WSE is enabled, all the constant speed pumps are commanded on.
</li>
<li>
When fully mechanical cooling (FMC) mode is enabled, the number of running constant speed pumps
equals to the number of running chillers.
</li>
</ul>
</html>",     revisions="<html>
<ul>
<li>
September 11, 2017, by Michael Wetter:<br/>
Revised switch that selects the operation mode for
<a href=\"https://github.com/lbl-srg/modelica-buildings/issues/921\">issue 921</a>
</li>
<li>
September 2, 2017, by Michael Wetter:<br/>
Changed implementation to use
<a href=\"modelica://FaultInjection.Experimental.SystemLevelFaults.Types.CoolingModes\">
FaultInjection.Experimental.SystemLevelFaults.Types.CoolingModes</a>.
</li>
<li>
July 30, 2017, by Yangyang Fu:<br/>
First implementation.
</li>
</ul>
</html>"));
    end ConstantSpeedPumpStage;

    model CoolingTowerSpeed "Controller for the fan speed in cooling towers"
      extends Modelica.Blocks.Icons.Block;

      parameter Modelica.Blocks.Types.SimpleController controllerType=
        Modelica.Blocks.Types.SimpleController.PID
        "Type of controller"
        annotation(Dialog(tab="Controller"));
      parameter Real k(min=0, unit="1") = 1
        "Gain of controller"
        annotation(Dialog(tab="Controller"));
      parameter Modelica.SIunits.Time Ti(min=Modelica.Constants.small)=0.5
        "Time constant of integrator block"
         annotation (Dialog(enable=
              (controllerType == Modelica.Blocks.Types.SimpleController.PI or
              controllerType == Modelica.Blocks.Types.SimpleController.PID),tab="Controller"));
      parameter Modelica.SIunits.Time Td(min=0)=0.1
        "Time constant of derivative block"
         annotation (Dialog(enable=
              (controllerType == Modelica.Blocks.Types.SimpleController.PD or
              controllerType == Modelica.Blocks.Types.SimpleController.PID),tab="Controller"));
      parameter Real yMax(start=1)=1
       "Upper limit of output"
        annotation(Dialog(tab="Controller"));
      parameter Real yMin=0
       "Lower limit of output"
        annotation(Dialog(tab="Controller"));
      parameter Boolean reverseAction = true
        "Set to true for throttling the water flow rate through a cooling coil controller"
        annotation(Dialog(tab="Controller"));
      Modelica.Blocks.Interfaces.RealInput TCHWSupSet(
        final quantity="ThermodynamicTemperature",
        final unit="K",
        displayUnit="degC") "Chilled water supply temperature setpoint"
        annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
      Modelica.Blocks.Interfaces.RealInput TCWSupSet(
        final quantity="ThermodynamicTemperature",
        final unit="K",
        displayUnit="degC") "Condenser water supply temperature setpoint"
        annotation (Placement(transformation(extent={{-140,60},{-100,100}})));
      Modelica.Blocks.Interfaces.RealInput TCHWSup(
        final quantity="ThermodynamicTemperature",
        final unit="K",
        displayUnit="degC") "Chilled water supply temperature " annotation (
          Placement(transformation(
            extent={{-20,-20},{20,20}},
            origin={-120,-74}), iconTransformation(extent={{-140,-100},{-100,-60}})));
      Modelica.Blocks.Interfaces.RealInput TCWSup(
        final quantity="ThermodynamicTemperature",
        final unit="K",
        displayUnit="degC") "Condenser water supply temperature " annotation (
          Placement(transformation(
            extent={{20,20},{-20,-20}},
            rotation=180,
            origin={-120,-40})));
      Modelica.Blocks.Interfaces.RealOutput y
        "Speed signal for cooling tower fans"
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));

      Modelica.Blocks.Sources.Constant uni(k=1) "Unit"
        annotation (Placement(transformation(extent={{-10,70},{10,90}})));
      Modelica.Blocks.Sources.BooleanExpression pmcMod(
        y= cooMod == Integer(FiveZone.Types.CoolingModes.PartialMechanical))
        "Partially mechanical cooling mode"
        annotation (Placement(transformation(extent={{-8,-10},{12,10}})));

      Modelica.Blocks.Interfaces.IntegerInput cooMod
        "Cooling mode signal, integer value of
    Buildings.Applications.DataCenters.Types.CoolingMode"
        annotation (Placement(transformation(extent={{-140,20},{-100,60}})));
      Buildings.Controls.Continuous.LimPID conPID(
        controllerType=controllerType,
        k=k,
        Ti=Ti,
        Td=Td,
        yMax=yMax,
        yMin=yMin,
        reverseAction=reverseAction,
        reset=Buildings.Types.Reset.Parameter,
        y_reset=0)
        "PID controller"
        annotation (Placement(transformation(extent={{-10,-50},{10,-30}})));
      Modelica.Blocks.Math.IntegerToBoolean fmcMod(threshold=Integer(FiveZone.Types.CoolingModes.FullMechanical))
        "Fully mechanical cooling mode"
        annotation (Placement(transformation(extent={{-90,30},{-70,50}})));

      Modelica.Blocks.Sources.BooleanExpression offMod(y=cooMod == Integer(FiveZone.Types.CoolingModes.Off))
        "off mode" annotation (Placement(transformation(extent={{30,22},{50,42}})));
      Modelica.Blocks.Sources.Constant off(k=0) "zero"
        annotation (Placement(transformation(extent={{30,54},{50,74}})));
      Buildings.Controls.OBC.CDL.Integers.LessThreshold notOff(threshold=Integer(FiveZone.Types.CoolingModes.Off))
        annotation (Placement(transformation(extent={{-88,-100},{-68,-80}})));
    protected
      Modelica.Blocks.Logical.Switch swi1
        "Switch 1"
        annotation (Placement(transformation(extent={{-46,30},{-26,50}})));
      Modelica.Blocks.Logical.Switch swi2
        "Switch 2"
        annotation (Placement(transformation(extent={{-10,-10},{10,10}},
            origin={-34,-60})));
      Modelica.Blocks.Logical.Switch swi3
        "Switch 3"
        annotation (Placement(transformation(extent={{-10,-10},{10,10}},
            origin={42,0})));

      Modelica.Blocks.Logical.Switch swi4
        "Switch 3"
        annotation (Placement(transformation(extent={{-10,-10},{10,10}},
            origin={80,32})));
    equation
      connect(TCWSupSet, swi1.u1)
        annotation (Line(points={{-120,80},{-58,80},{-58,48},{-48,48}},
                             color={0,0,127}));
      connect(TCHWSupSet, swi1.u3)
        annotation (Line(points={{-120,0},{-58,0},{-58,32},{-48,32}},
                             color={0,0,127}));
      connect(swi1.y, conPID.u_s)
        annotation (Line(points={{-25,40},{-20,40},{-20,-40},{-12,-40}},
                     color={0,0,127}));
      connect(fmcMod.y, swi2.u2)
        annotation (Line(points={{-69,40},{-64,40},{-64,-60},{-46,-60}},
                          color={255,0,255}));
      connect(TCWSup, swi2.u1)
        annotation (Line(points={{-120,-40},{-60,-40},{-60,-52},{-46,-52}},
                          color={0,0,127}));
      connect(swi2.y, conPID.u_m)
        annotation (Line(points={{-23,-60},{0,-60},{0,-52}},   color={0,0,127}));
      connect(pmcMod.y, swi3.u2)
        annotation (Line(points={{13,0},{30,0}},          color={255,0,255}));
      connect(uni.y, swi3.u1)
        annotation (Line(points={{11,80},{20,80},{20,8},{30,8}}, color={0,0,127}));
      connect(fmcMod.y, swi1.u2)
        annotation (Line(points={{-69,40},{-48,40}},
                         color={255,0,255}));
      connect(cooMod, fmcMod.u)
        annotation (Line(points={{-120,40},{-92,40}},
                    color={255,127,0}));
      connect(conPID.y, swi3.u3) annotation (Line(points={{11,-40},{20,-40},{20,-8},
              {30,-8}}, color={0,0,127}));
      connect(offMod.y, swi4.u2)
        annotation (Line(points={{51,32},{68,32}}, color={255,0,255}));
      connect(off.y, swi4.u1) annotation (Line(points={{51,64},{60,64},{60,40},{68,
              40}}, color={0,0,127}));
      connect(swi3.y, swi4.u3)
        annotation (Line(points={{53,0},{60,0},{60,24},{68,24}}, color={0,0,127}));
      connect(swi4.y, y) annotation (Line(points={{91,32},{96,32},{96,0},{110,0}},
            color={0,0,127}));
      connect(cooMod, notOff.u) annotation (Line(points={{-120,40},{-96,40},{-96,
              -90},{-90,-90}}, color={255,127,0}));
      connect(TCHWSup, swi2.u3) annotation (Line(points={{-120,-74},{-60,-74},{-60,
              -68},{-46,-68}}, color={0,0,127}));
      connect(notOff.y, conPID.trigger)
        annotation (Line(points={{-66,-90},{-8,-90},{-8,-52}}, color={255,0,255}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false, extent={{-100,
                -100},{100,80}})),    Documentation(info="<html>
<p>This model describes a simple cooling tower speed controller for
a chilled water system with integrated waterside economizers.
</p>
<p>The control logics are described in the following:</p>
<ul>
<li>When the system is in Fully Mechanical Cooling (FMC) mode,
the cooling tower fan speed is controlled to maintain the condener water supply temperature (CWST)
at or around the setpoint.
</li>
<li>When the system is in Partially Mechanical Cooling (PMC) mode,
the cooling tower fan speed is set as 100% to make condenser water
as cold as possible and maximize the waterside economzier output.
</li>
<li>When the system is in Free Cooling (FC) mode,
the cooling tower fan speed is controlled to maintain the chilled water supply temperature (CHWST)
at or around its setpoint.
</li>
</ul>
</html>",     revisions="<html>
<ul>
<li>
July 30, 2017, by Yangyang Fu:<br/>
First implementation.
</li>
</ul>
</html>"));
    end CoolingTowerSpeed;

    model TemperatureDifferentialPressureReset
      "CHWST and CHW DP reset control for chillers"
      extends Modelica.Blocks.Icons.Block;
      parameter Modelica.SIunits.Time samplePeriod=120 "Sample period of component";
      parameter Real uTri=0 "Value to triggering the request for actuator";
      parameter Real yEqu0=0 "y setpoint when equipment starts";
      parameter Real yDec=-0.03 "y decrement (must be negative)";
      parameter Real yInc=0.03 "y increment (must be positive)";
      parameter Real x1=0.5 "First interval [x0, x1] and second interval (x1, x2]"
      annotation(Dialog(tab="Pressure and temperature reset points"));
      parameter Modelica.SIunits.Pressure dpMin = 100 "dpmin"
      annotation(Dialog(tab="Pressure and temperature reset points"));
      parameter Modelica.SIunits.Pressure dpMax =  300 "dpmax"
      annotation(Dialog(tab="Pressure and temperature reset points"));
      parameter Modelica.SIunits.ThermodynamicTemperature TMin=273.15+5.56 "Tchi,min"
      annotation(Dialog(tab="Pressure and temperature reset points"));
      parameter Modelica.SIunits.ThermodynamicTemperature TMax = 273.15+22 "Tchi,max"
      annotation(Dialog(tab="Pressure and temperature reset points"));
      parameter Modelica.SIunits.Time startTime=0 "First sample time instant";

      Modelica.Blocks.Interfaces.RealInput u
        "Input signall, such as dT, or valve position"
        annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
      FiveZone.PrimarySideControl.BaseClasses.LinearPiecewiseTwo linPieTwo(
        x0=0,
        x1=x1,
        x2=1,
        y10=dpMin,
        y11=dpMax,
        y20=TMax,
        y21=TMin) "Calculation of two piecewise linear functions"
        annotation (Placement(transformation(extent={{20,-10},{40,10}})));

      Modelica.Blocks.Interfaces.RealOutput dpSet(
        final quantity="Pressure",
        final unit = "Pa") "DP setpoint"
        annotation (Placement(transformation(extent={{100,40},{120,60}}),
            iconTransformation(extent={{100,40},{120,60}})));
      Modelica.Blocks.Interfaces.RealOutput TSet(
        final quantity="ThermodynamicTemperature",
        final unit="K")
        "CHWST"
        annotation (Placement(transformation(extent={{100,-60},{120,-40}}),
            iconTransformation(extent={{100,-60},{120,-40}})));
      FiveZone.PrimarySideControl.BaseClasses.TrimAndRespond triAndRes(
        samplePeriod=samplePeriod,
        startTime=startTime,
        uTri=uTri,
        yEqu0=yEqu0,
        yDec=yDec,
        yInc=yInc)
        annotation (Placement(transformation(extent={{-40,-10},{-20,10}})));

      Modelica.Blocks.Logical.Switch swi1 "Switch"
        annotation (Placement(transformation(extent={{60,70},{80,90}})));
      Modelica.Blocks.Sources.RealExpression dpSetIni(y=dpMin)
        "Initial dp setpoint"
        annotation (Placement(transformation(extent={{-20,80},{0,100}})));
      Modelica.Blocks.Sources.RealExpression TSetIni(y=TMin)
        "Initial temperature setpoint"
        annotation (Placement(transformation(extent={{-20,-88},{0,-68}})));
      Modelica.Blocks.Logical.Switch swi2 "Switch"
        annotation (Placement(transformation(extent={{60,-80},{80,-60}})));
      Modelica.Blocks.Interfaces.IntegerInput uOpeMod
        "Cooling mode in WSEControls.Type.OperationaModes" annotation (Placement(
            transformation(extent={{-140,40},{-100,80}}),  iconTransformation(
              extent={{-140,40},{-100,80}})));
      Buildings.Controls.OBC.CDL.Integers.GreaterThreshold intGreThr(threshold=
            Integer(FiveZone.Types.CoolingModes.FreeCooling))
        annotation (Placement(transformation(extent={{-60,50},{-40,70}})));
      Buildings.Controls.OBC.CDL.Logical.And and2
        annotation (Placement(transformation(extent={{-10,50},{10,70}})));
      Buildings.Controls.OBC.CDL.Integers.LessThreshold intLesThr(threshold=Integer(FiveZone.Types.CoolingModes.Off))
        annotation (Placement(transformation(extent={{-60,18},{-40,38}})));
    equation

      connect(triAndRes.y, linPieTwo.u)
        annotation (Line(points={{-19,0},{18,0}}, color={0,0,127}));
      connect(uOpeMod, intGreThr.u)
        annotation (Line(points={{-120,60},{-62,60}}, color={255,127,0}));
      connect(linPieTwo.y[1], swi1.u1) annotation (Line(points={{41,-0.5},{48,-0.5},
              {48,88},{58,88}}, color={0,0,127}));
      connect(dpSetIni.y, swi1.u3) annotation (Line(points={{1,90},{46,90},{46,72},
              {58,72}}, color={0,0,127}));
      connect(linPieTwo.y[2], swi2.u1) annotation (Line(points={{41,0.5},{52,0.5},{
              52,-62},{58,-62}}, color={0,0,127}));
      connect(TSetIni.y, swi2.u3)
        annotation (Line(points={{1,-78},{58,-78}}, color={0,0,127}));
      connect(swi1.y, dpSet) annotation (Line(points={{81,80},{90,80},{90,50},{110,
              50}}, color={0,0,127}));
      connect(swi2.y, TSet) annotation (Line(points={{81,-70},{90,-70},{90,-50},{
              110,-50}}, color={0,0,127}));
      connect(u, triAndRes.u)
        annotation (Line(points={{-120,0},{-42,0}}, color={0,0,127}));
      connect(uOpeMod, intLesThr.u) annotation (Line(points={{-120,60},{-80,60},{
              -80,28},{-62,28}}, color={255,127,0}));
      connect(intGreThr.y, and2.u1)
        annotation (Line(points={{-38,60},{-12,60}}, color={255,0,255}));
      connect(intLesThr.y, and2.u2) annotation (Line(points={{-38,28},{-28,28},{-28,
              52},{-12,52}}, color={255,0,255}));
      connect(and2.y, swi1.u2) annotation (Line(points={{12,60},{34,60},{34,80},{58,
              80}}, color={255,0,255}));
      connect(and2.y, swi2.u2) annotation (Line(points={{12,60},{46,60},{46,-70},{
              58,-70}}, color={255,0,255}));
      annotation (defaultComponentName="temDifPreRes",
        Documentation(info="<html>
<p>This model describes a chilled water supply temperature setpoint and differential pressure setpoint reset control. In this logic, it is to first increase the different pressure, <i>&Delta;p</i>, of the chilled water loop to increase the mass flow rate. If <i>&Delta;p</i> reaches the maximum value and further cooling is still needed, the chiller remperature setpoint, <i>T<sub>chi,set</i></sub>, is reduced. If there is too much cooling, the <i>T<sub>chi,set</i></sub> and <i>&Delta;p</i> will be changed in the reverse direction. </p>
<p>The model implements a discrete time trim and respond logic as follows: </p>
<ul>
<li>A cooling request is triggered if the input signal, <i>y</i>, is larger than 0. <i>y</i> is the difference between the actual and set temperature of the suppuly air to the data center room.</li>
<li>The request is sampled every 2 minutes. If there is a cooling request, the control signal <i>u</i> is increased by <i>0.03</i>, where <i>0 &le; u &le; 1</i>. If there is no cooling request, <i>u</i> is decreased by <i>0.03</i>. </li>
</ul>
<p>The control signal <i>u</i> is converted to setpoints for <i>&Delta;p</i> and <i>T<sub>chi,set</i></sub> as follows: </p>
<ul>
<li>If <i>u &isin; [0, x]</i> then <i>&Delta;p = &Delta;p<sub>min</sub> + u &nbsp;(&Delta;p<sub>max</sub>-&Delta;p<sub>min</sub>)/x</i> and <i>T = T<sub>max</i></sub></li>
<li>If <i>u &isin; (x, 1]</i> then <i>&Delta;p = &Delta;p<sub>max</i></sub> and <i>T = T<sub>max</sub> - (u-x)&nbsp;(T<sub>max</sub>-T<sub>min</sub>)/(1-x) </i></li>
</ul>
<p>where <i>&Delta;p<sub>min</i></sub> and <i>&Delta;p<sub>max</i></sub> are minimum and maximum values for <i>&Delta;p</i>, and <i>T<sub>min</i></sub> and <i>T<sub>max</i></sub> are the minimum and maximum values for <i>T<sub>chi,set</i></sub>. </p>
<p>Note that we deactivate the trim and response when the chillers are off.</p>

<h4>Reference</h4>
<p>Stein, J. (2009). Waterside Economizing in Data Centers: Design and Control Considerations. ASHRAE Transactions, 115(2), 192-200.</p>
<p>Taylor, S.T. (2007). Increasing Efficiency with VAV System Static Pressure Setpoint Reset. ASHRAE Journal, June, 24-32. </p>
</html>",     revisions="<html>
<ul>
<li><i>December 19, 2018</i> by Yangyang Fu:<br/>
        Deactivate reset when chillers are off.
</li>
<li><i>June 23, 2018</i> by Xing Lu:<br/>
        First implementation.
</li>
</ul>
</html>"));
    end TemperatureDifferentialPressureReset;

    model PlantRequest "Plant request control"
      extends Modelica.Blocks.Icons.Block;

      Buildings.Controls.OBC.CDL.Interfaces.IntegerOutput yPlaReq
        "Plant request" annotation (Placement(transformation(extent={{100,-10},{120,
                10}}), iconTransformation(extent={{100,-10},{120,10}})));
      Buildings.Controls.OBC.CDL.Interfaces.RealInput uPlaVal(
        min=0,
        max=1,
        final unit="1") "Cooling or Heating valve position"
    annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
      Buildings.Controls.OBC.CDL.Continuous.Hysteresis hys(final uHigh=uHigh,
          final uLow=uLow)
        "Check if valve position is greater than 0.95"
        annotation (Placement(transformation(extent={{-70,-10},{-50,10}})));
      parameter Real uLow=0.1 "if y=true and u<uLow, switch to y=false";
      parameter Real uHigh=0.95 "if y=false and u>uHigh, switch to y=true";
    protected
      Buildings.Controls.OBC.CDL.Continuous.Sources.Constant onePlaReq(final k=1)
        "Constant 1"
        annotation (Placement(transformation(extent={{-10,20},{10,40}})));
      Buildings.Controls.OBC.CDL.Continuous.Sources.Constant zerPlaReq(final k=0) "Constant 0"
        annotation (Placement(transformation(extent={{-10,-40},{10,-20}})));
      Buildings.Controls.OBC.CDL.Logical.Switch swi "Output 0 or 1 request "
        annotation (Placement(transformation(extent={{30,-10},{50,10}})));
      Buildings.Controls.OBC.CDL.Conversions.RealToInteger reaToInt "Convert real to integer value"
        annotation (Placement(transformation(extent={{70,-10},{90,10}})));
    equation
      connect(reaToInt.y, yPlaReq) annotation (Line(points={{92,0},{96,0},{96,0},{
              110,0}},
                   color={255,127,0}));
      connect(swi.y, reaToInt.u)
        annotation (Line(points={{52,0},{68,0}}, color={0,0,127}));
      connect(onePlaReq.y, swi.u1)
        annotation (Line(points={{12,30},{20,30},{20,8},{28,8}}, color={0,0,127}));
      connect(zerPlaReq.y, swi.u3) annotation (Line(points={{12,-30},{20,-30},{20,
              -8},{28,-8}},
                        color={0,0,127}));
      connect(hys.y, swi.u2)
        annotation (Line(points={{-48,0},{28,0}}, color={255,0,255}));
      connect(uPlaVal, hys.u)
        annotation (Line(points={{-120,0},{-72,0}}, color={0,0,127}));
      annotation (Documentation(info="<html>
<p>This module calculates the plant request number based on the valve position: </p>
<p><br><b>Chiller Plant Requests</b>. Send the chiller plant that serves the system a Chiller Plant Request as follows: </p>
<ol>
<li>If the CHW valve position is greater than 95&percnt;, send 1 Request until the CHW valve position is less than 10&percnt; </li>
<li>Else if the CHW valve position is less than 95&percnt;, send 0 Requests. </li>
</ol>
<p><b>Hot Water Plant Requests: </b>Send the heating hot water plant that serves the AHU a Hot Water Plant Request as follows: </p>
<ol>
<li>If the HW valve position is greater than 95&percnt;, send 1 Request until the HW valve position is less than 10&percnt; </li>
<li>Else if the HW valve position is less than 95&percnt;, send 0 Requests. </li>
</ol>
</html>",     revisions="<html>
<ul>
<li>Sep 1, 2020, by Xing Lu:<br>First implementation. </li>
</ul>
</html>"));
    end PlantRequest;

    model BoilerPlantEnableDisable
      extends ChillerPlantEnableDisable(con1(condition=yPlaReq > numIgn and TOut <
              TOutPla and ySupFan and offTim.y >= shoCycTim), con2(condition=(
              lesEquReq.y >= plaReqTim and onTim.y >= shoCycTim and lesEquSpe.y >=
              plaReqTim) or ((TOut > TOutPla - 1 or not ySupFan) and onTim.y >=
              shoCycTim), waitTime=0));
      annotation (Documentation(info="<html>
<p>This is a boiler plant enable disable control that works as follows: </p>
<p>Enable the plant in the lowest stage when the plant has been disabled for at least 15 minutes and: </p>
<ol>
<li>Number of Hot Water Plant Requests &gt; I (I = Ignores shall default to 0, adjustable), and </li>
<li>OAT&lt;H-LOT, and </li>
<li>The boiler plant enable schedule is active. </li>
</ol>
<p>Disable the plant when it has been enabled for at least 15 minutes and: </p>
<ol>
<li>Number of Hot Water Plant Requests &le; I for 3 minutes, or </li>
<li>OAT&gt;H-LOT-1&deg;F, or </li>
<li>The boiler plant enable schedule is inactive. </li>
</ol>
<p>In the above logic, OAT is the outdoor air temperature, CH-LOT is the chiller plant lockout air temperature, H-LOT is the heating plant lockout air temperature.</p>
</html>"));
    end BoilerPlantEnableDisable;

    model MinimumFlowBypassValve "Minimum flow bypass valve control"
      extends Modelica.Blocks.Icons.Block;

      Buildings.Controls.OBC.CDL.Interfaces.RealInput m_flow(final quantity=
            "MassFlowRate", final unit="kg/s") "Water mass flow rate measurement"
        annotation (Placement(transformation(extent={{-140,10},{-100,50}}),
            iconTransformation(extent={{-20,-20},{20,20}}, origin={-120,30})));
      Modelica.Blocks.Sources.RealExpression m_flow_min(y=m_flow_minimum)
        "Design minimum water flow rate"
        annotation (Placement(transformation(extent={{-80,48},{-60,68}})));
      Buildings.Controls.OBC.CDL.Continuous.LimPID conPID(
        controllerType=controllerType,
        k=k,
        Ti=Ti,
        Td=Td,
        reset=Buildings.Types.Reset.Parameter,
        y_reset=0) annotation (Placement(transformation(extent={{-10,60},{10,80}})));
      Modelica.Blocks.Interfaces.RealOutput y
        annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      parameter Modelica.SIunits.MassFlowRate m_flow_minimum=0.1 "Design minimum water mass flow rate";
     // Controller
      parameter Modelica.Blocks.Types.SimpleController controllerType=
        Modelica.Blocks.Types.SimpleController.PI
        "Type of controller";
      parameter Real k(min=0, unit="1") = 0.1
        "Gain of controller";
      parameter Modelica.SIunits.Time Ti(min=Modelica.Constants.small)=60
        "Time constant of integrator block"
         annotation (Dialog(enable=
              (controllerType == Modelica.Blocks.Types.SimpleController.PI or
              controllerType == Modelica.Blocks.Types.SimpleController.PID)));
      parameter Modelica.SIunits.Time Td(min=0)=0.1
        "Time constant of derivative block"
         annotation (Dialog(enable=
             (controllerType == Modelica.Blocks.Types.SimpleController.PD or
              controllerType == Modelica.Blocks.Types.SimpleController.PID)));

      Buildings.Controls.OBC.CDL.Interfaces.BooleanInput yPla "Plant on/off"
        annotation (Placement(transformation(extent={{-140,-70},{-100,-30}}),
            iconTransformation(extent={{-140,-50},{-100,-10}})));
      Modelica.Blocks.Sources.RealExpression dm(y=m_flow - m_flow_minimum)
        "Delta mass flowrate"
        annotation (Placement(transformation(extent={{-92,-20},{-72,0}})));
      Buildings.Controls.OBC.CDL.Continuous.Hysteresis hys(uLow=0, uHigh=0.1,
        y(start=false))
        annotation (Placement(transformation(extent={{-60,-20},{-40,0}})));
      Modelica.Blocks.Logical.And and1
        annotation (Placement(transformation(extent={{0,-40},{20,-20}})));
    protected
      Buildings.Controls.OBC.CDL.Logical.Switch swi "Output 0 or 1 request "
        annotation (Placement(transformation(extent={{54,-10},{74,10}})));
      Buildings.Controls.OBC.CDL.Continuous.Sources.Constant zer(final k=0)
        "Constant 0"
        annotation (Placement(transformation(extent={{20,30},{40,50}})));
    equation
      connect(m_flow_min.y, conPID.u_s)
        annotation (Line(points={{-59,58},{-36,58},{-36,70},{-12,70}},
                                                   color={0,0,127}));
      connect(conPID.u_m, m_flow)
        annotation (Line(points={{0,58},{0,30},{-120,30}}, color={0,0,127}));
      connect(swi.y, y) annotation (Line(points={{76,0},{110,0}}, color={0,0,127}));
      connect(dm.y, hys.u)
        annotation (Line(points={{-71,-10},{-62,-10}}, color={0,0,127}));
      connect(conPID.y, swi.u3) annotation (Line(points={{12,70},{16,70},{16,-8},{52,
              -8}},    color={0,0,127}));
      connect(zer.y, swi.u1)
        annotation (Line(points={{42,40},{44,40},{44,8},{52,8}}, color={0,0,127}));
      connect(yPla, and1.u2) annotation (Line(points={{-120,-50},{-20,-50},{-20,-38},
              {-2,-38}}, color={255,0,255}));
      connect(hys.y, and1.u1) annotation (Line(points={{-38,-10},{-20,-10},{-20,
              -30},{-2,-30}},
                         color={255,0,255}));
      connect(and1.y, swi.u2) annotation (Line(points={{21,-30},{40,-30},{40,0},{52,
              0}}, color={255,0,255}));
      connect(hys.y, conPID.trigger)
        annotation (Line(points={{-38,-10},{-6,-10},{-6,58}}, color={255,0,255}));
      annotation (Documentation(info="<html>
<p>The bypass valve PID loop is enabled when the plant is on. When enabled, the bypass valve loop starts with the valve 0&percnt; open. It is closed when the plant is off. </p>
</html>",     revisions="<html>
<ul>
<li>Sep 1, 2020, by Xing Lu:<br>First implementation. </li>
</ul>
</html>"));
    end MinimumFlowBypassValve;

    model HotWaterTemperatureReset "Hot Water Temperature Reset Control"

      parameter Real uHigh=0.95 "if y=false and u>uHigh, switch to y=true";
      parameter Real uLow=0.85 "if y=true and u<uLow, switch to y=false";
      Buildings.Controls.OBC.CDL.Interfaces.RealInput uPlaHeaVal
        "Heating valve position"
        annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
      parameter Real iniSet(
        final unit="K",
        final displayUnit="degC",
        final quantity="ThermodynamicTemperature") = maxSet
        "Initial setpoint"
        annotation (Dialog(group="Trim and respond for pressure setpoint"));
      parameter Real minSet(
        final unit="K",
        final displayUnit="degC",
        final quantity="ThermodynamicTemperature") = 273.15 + 32.2
        "Minimum setpoint"
        annotation (Dialog(group="Trim and respond for pressure setpoint"));
      parameter Real maxSet(
        final unit="K",
        final displayUnit="degC",
        final quantity="ThermodynamicTemperature") = 273.15 + 45
        "Maximum setpoint"
        annotation (Dialog(group="Trim and respond for pressure setpoint"));
      parameter Real delTim(
        final unit="s",
        final quantity="Time")= 600
       "Delay time after which trim and respond is activated"
        annotation (Dialog(group="Trim and respond for pressure setpoint"));
      parameter Real samplePeriod(
        final unit="s",
        final quantity="Time") = 300  "Sample period"
        annotation (Dialog(group="Trim and respond for pressure setpoint"));
      parameter Integer numIgnReq = 0
        "Number of ignored requests"
        annotation (Dialog(group="Trim and respond for pressure setpoint"));
      parameter Real triAmo(
        final unit="K",
        final displayUnit="K",
        final quantity="TemperatureDifference") = -1
        "Trim amount"
        annotation (Dialog(group="Trim and respond for pressure setpoint"));
      parameter Real resAmo(
        final unit="K",
        final displayUnit="K",
        final quantity="TemperatureDifference") = 1.5
        "Response amount"
        annotation (Dialog(group="Trim and respond for pressure setpoint"));
      parameter Real maxRes(
        final unit="K",
        final displayUnit="K",
        final quantity="TemperatureDifference") = 4
        "Maximum response per time interval (same sign as resAmo)"
        annotation (Dialog(group="Trim and respond for pressure setpoint"));
      Buildings.Controls.OBC.ASHRAE.G36_PR1.Generic.SetPoints.TrimAndRespond staTBoiSupSetRes(
        final iniSet=iniSet,
        final minSet=minSet,
        final maxSet=maxSet,
        final delTim=delTim,
        final samplePeriod=samplePeriod,
        final numIgnReq=numIgnReq,
        final triAmo=triAmo,
        final resAmo=resAmo,
        final maxRes=maxRes)
        "Static pressure setpoint reset using trim and respond logic"
        annotation (Placement(transformation(extent={{-40,-2},{-20,18}})));

      FiveZone.Controls.PlantRequest plaReq(uLow=uLow, uHigh=uHigh)
        annotation (Placement(transformation(extent={{-80,-10},{-60,10}})));
      Buildings.Controls.OBC.CDL.Interfaces.BooleanInput uDevSta
        "On/Off status of the associated device"
        annotation (Placement(transformation(extent={{-140,54},{-100,94}})));
      Buildings.Controls.OBC.CDL.Interfaces.RealOutput TSupBoi(
        final unit="K",
        final displayUnit="degC",
        final quantity="ThermodynamicTemperature")
        "Setpoint for boiler supply water temperature"
        annotation (Placement(transformation(extent={{100,-12},{140,28}}),
            iconTransformation(extent={{100,-20},{140,20}})));
    protected
      Buildings.Controls.OBC.CDL.Discrete.FirstOrderHold firOrdHol(final
          samplePeriod=samplePeriod)
        "Extrapolation through the values of the last two sampled input signals"
        annotation (Placement(transformation(extent={{10,-2},{30,18}})));
    equation
      connect(plaReq.uPlaVal, uPlaHeaVal)
        annotation (Line(points={{-82,0},{-120,0}}, color={0,0,127}));
      connect(plaReq.yPlaReq, staTBoiSupSetRes.numOfReq)
        annotation (Line(points={{-59,0},{-42,0}}, color={255,127,0}));
      connect(staTBoiSupSetRes.uDevSta, uDevSta) annotation (Line(points={{-42,16},{
              -50,16},{-50,74},{-120,74}}, color={255,0,255}));
      connect(staTBoiSupSetRes.y, firOrdHol.u)
        annotation (Line(points={{-18,8},{8,8}}, color={0,0,127}));
      connect(firOrdHol.y, TSupBoi)
        annotation (Line(points={{32,8},{120,8}}, color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false), graphics={
                                    Rectangle(
              extent={{-100,-100},{100,100}},
              lineColor={0,0,127},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
                                            Text(
            extent={{-152,146},{148,106}},
            textString="%name",
            lineColor={0,0,255})}),                                  Diagram(
            coordinateSystem(preserveAspectRatio=false)),
        Documentation(info="<html>
<p>Hot water supply temperature setpoint shall be reset using Trim &amp; Respond logic using following parameters as a starting point: </p>
<table cellspacing=\"2\" cellpadding=\"0\" border=\"1\"><tr>
<td><p align=\"center\"><h4>Variable </h4></p></td>
<td><p align=\"center\"><h4>Value </h4></p></td>
<td><p align=\"center\"><h4>Definition </h4></p></td>
</tr>
<tr>
<td><p>Device</p></td>
<td><p>HW Loop</p></td>
<td><p>Associated device</p></td>
</tr>
<tr>
<td><p>SP0</p></td>
<td><p><span style=\"font-family: Courier New;\">iniSet</span></p></td>
<td><p>Initial setpoint</p></td>
</tr>
<tr>
<td><p>SPmin</p></td>
<td><p><span style=\"font-family: Courier New;\">minSet</span></p></td>
<td><p>Minimum setpoint</p></td>
</tr>
<tr>
<td><p>SPmax</p></td>
<td><p><span style=\"font-family: Courier New;\">maxSet</span></p></td>
<td><p>Maximum setpoint</p></td>
</tr>
<tr>
<td><p>Td</p></td>
<td><p><span style=\"font-family: Courier New;\">delTim</span></p></td>
<td><p>Delay timer</p></td>
</tr>
<tr>
<td><p>T</p></td>
<td><p><span style=\"font-family: Courier New;\">samplePeriod</span></p></td>
<td><p>Time step</p></td>
</tr>
<tr>
<td><p>I</p></td>
<td><p><span style=\"font-family: Courier New;\">numIgnReq</span></p></td>
<td><p>Number of ignored requests</p></td>
</tr>
<tr>
<td><p>R</p></td>
<td><p><span style=\"font-family: Courier New;\">uZonPreResReq</span></p></td>
<td><p>Number of requests</p></td>
</tr>
<tr>
<td><p>SPtrim</p></td>
<td><p><span style=\"font-family: Courier New;\">triAmo</span></p></td>
<td><p>Trim amount</p></td>
</tr>
<tr>
<td><p>SPres</p></td>
<td><p><span style=\"font-family: Courier New;\">resAmo</span></p></td>
<td><p>Respond amount</p></td>
</tr>
<tr>
<td><p>SPres_max</p></td>
<td><p><span style=\"font-family: Courier New;\">maxRes</span></p></td>
<td><p>Maximum response per time interval</p></td>
</tr>
</table>
</html>"));
    end HotWaterTemperatureReset;

    model TrimResponse "Trim and respond"
      extends Modelica.Blocks.Icons.Block;
      parameter Modelica.SIunits.Time samplePeriod=120 "Sample period of component";
      parameter Real uTri=0 "Value to triggering the request for actuator";
      parameter Real yEqu0=0 "y setpoint when equipment starts";
      parameter Real yDec=-0.03 "y decrement (must be negative)";
      parameter Real yInc=0.03 "y increment (must be positive)";
      parameter Real x1=0.5 "First interval [x0, x1] and second interval (x1, x2]"
      annotation(Dialog(tab="Pressure and temperature reset points"));
      parameter Modelica.SIunits.Pressure dpMin = 100 "dpmin"
      annotation(Dialog(tab="Pressure and temperature reset points"));
      parameter Modelica.SIunits.Pressure dpMax =  300 "dpmax"
      annotation(Dialog(tab="Pressure and temperature reset points"));
      parameter Modelica.SIunits.ThermodynamicTemperature TMin=273.15+32 "Tchi,min"
      annotation(Dialog(tab="Pressure and temperature reset points"));
      parameter Modelica.SIunits.ThermodynamicTemperature TMax = 273.15+45 "Tchi,max"
      annotation(Dialog(tab="Pressure and temperature reset points"));
      parameter Modelica.SIunits.Time startTime=0 "First sample time instant";

      Modelica.Blocks.Interfaces.RealInput u
        "Input signall, such as dT, or valve position"
        annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
      FiveZone.PrimarySideControl.BaseClasses.LinearPiecewiseTwo linPieTwo(
        x0=0,
        x1=x1,
        x2=1,
        y10=dpMin,
        y11=dpMax,
        y20=TMin,
        y21=TMax) "Calculation of two piecewise linear functions"
        annotation (Placement(transformation(extent={{20,-10},{40,10}})));

      Modelica.Blocks.Interfaces.RealOutput dpSet(
        final quantity="Pressure",
        final unit = "Pa") "DP setpoint"
        annotation (Placement(transformation(extent={{100,40},{120,60}}),
            iconTransformation(extent={{100,40},{120,60}})));
      Modelica.Blocks.Interfaces.RealOutput TSet(
        final quantity="ThermodynamicTemperature",
        final unit="K")
        "CHWST"
        annotation (Placement(transformation(extent={{100,-60},{120,-40}}),
            iconTransformation(extent={{100,-60},{120,-40}})));
      FiveZone.PrimarySideControl.BaseClasses.TrimAndRespond triAndRes(
        samplePeriod=samplePeriod,
        startTime=startTime,
        uTri=uTri,
        yEqu0=yEqu0,
        yDec=yDec,
        yInc=yInc)
        annotation (Placement(transformation(extent={{-40,-10},{-20,10}})));

      Buildings.Controls.OBC.CDL.Interfaces.BooleanInput uDevSta
        "On/Off status of the associated device"
        annotation (Placement(transformation(extent={{-140,50},{-100,90}}),
            iconTransformation(extent={{-180,10},{-100,90}})));
      Modelica.Blocks.Sources.RealExpression dpSetIni(y=dpMin)
        "Initial dp setpoint"
        annotation (Placement(transformation(extent={{-20,80},{0,100}})));
      Modelica.Blocks.Sources.RealExpression TSetIni(y=TMin)
        "Initial temperature setpoint"
        annotation (Placement(transformation(extent={{-20,-88},{0,-68}})));
      Modelica.Blocks.Logical.Switch swi2 "Switch"
        annotation (Placement(transformation(extent={{60,-80},{80,-60}})));
      Modelica.Blocks.Logical.Switch swi1 "Switch"
        annotation (Placement(transformation(extent={{60,70},{80,90}})));
    equation

      connect(triAndRes.y, linPieTwo.u)
        annotation (Line(points={{-19,0},{18,0}}, color={0,0,127}));
      connect(u, triAndRes.u)
        annotation (Line(points={{-120,0},{-42,0}}, color={0,0,127}));
      connect(uDevSta, swi1.u2) annotation (Line(points={{-120,70},{0,70},{0,80},{
              58,80}}, color={255,0,255}));
      connect(swi1.y, dpSet) annotation (Line(points={{81,80},{92,80},{92,50},{110,
              50}}, color={0,0,127}));
      connect(linPieTwo.y[1], swi1.u1) annotation (Line(points={{41,-0.5},{48,-0.5},
              {48,88},{58,88}}, color={0,0,127}));
      connect(dpSetIni.y, swi1.u3) annotation (Line(points={{1,90},{46,90},{46,72},
              {58,72}}, color={0,0,127}));
      connect(uDevSta, swi2.u2) annotation (Line(points={{-120,70},{0,70},{0,-70},{
              58,-70}}, color={255,0,255}));
      connect(TSetIni.y, swi2.u3)
        annotation (Line(points={{1,-78},{58,-78}}, color={0,0,127}));
      connect(linPieTwo.y[2], swi2.u1) annotation (Line(points={{41,0.5},{48,0.5},{
              48,-62},{58,-62}}, color={0,0,127}));
      connect(swi2.y, TSet) annotation (Line(points={{81,-70},{88,-70},{88,-50},{
              110,-50}}, color={0,0,127}));
      annotation (defaultComponentName="triRes",
        Documentation(info="<html>
<p>This model describes a chilled water supply temperature setpoint and differential pressure setpoint reset control. In this logic, it is to first increase the different pressure, <i>&Delta;p</i>, of the chilled water loop to increase the mass flow rate. If <i>&Delta;p</i> reaches the maximum value and further cooling is still needed, the chiller remperature setpoint, <i>T<sub>chi,set</i></sub>, is reduced. If there is too much cooling, the <i>T<sub>chi,set</i></sub> and <i>&Delta;p</i> will be changed in the reverse direction. </p>
<p>The model implements a discrete time trim and respond logic as follows: </p>
<ul>
<li>A cooling request is triggered if the input signal, <i>y</i>, is larger than 0. <i>y</i> is the difference between the actual and set temperature of the suppuly air to the data center room.</li>
<li>The request is sampled every 2 minutes. If there is a cooling request, the control signal <i>u</i> is increased by <i>0.03</i>, where <i>0 &le; u &le; 1</i>. If there is no cooling request, <i>u</i> is decreased by <i>0.03</i>. </li>
</ul>
<p>The control signal <i>u</i> is converted to setpoints for <i>&Delta;p</i> and <i>T<sub>chi,set</i></sub> as follows: </p>
<ul>
<li>If <i>u &isin; [0, x]</i> then <i>&Delta;p = &Delta;p<sub>min</sub> + u &nbsp;(&Delta;p<sub>max</sub>-&Delta;p<sub>min</sub>)/x</i> and <i>T = T<sub>max</i></sub></li>
<li>If <i>u &isin; (x, 1]</i> then <i>&Delta;p = &Delta;p<sub>max</i></sub> and <i>T = T<sub>max</sub> - (u-x)&nbsp;(T<sub>max</sub>-T<sub>min</sub>)/(1-x) </i></li>
</ul>
<p>where <i>&Delta;p<sub>min</i></sub> and <i>&Delta;p<sub>max</i></sub> are minimum and maximum values for <i>&Delta;p</i>, and <i>T<sub>min</i></sub> and <i>T<sub>max</i></sub> are the minimum and maximum values for <i>T<sub>chi,set</i></sub>. </p>
<p>Note that we deactivate the trim and response when the chillers are off.</p>

<h4>Reference</h4>
<p>Stein, J. (2009). Waterside Economizing in Data Centers: Design and Control Considerations. ASHRAE Transactions, 115(2), 192-200.</p>
<p>Taylor, S.T. (2007). Increasing Efficiency with VAV System Static Pressure Setpoint Reset. ASHRAE Journal, June, 24-32. </p>
</html>",     revisions="<html>
<ul>
<li><i>December 19, 2018</i> by Yangyang Fu:<br/>
        Deactivate reset when chillers are off.
</li>
<li><i>June 23, 2018</i> by Xing Lu:<br/>
        First implementation.
</li>
</ul>
</html>"));
    end TrimResponse;

    package Validation

      model ChillerPlantEnableDisable
        extends Modelica.Icons.Example;
        FiveZone.Controls.ChillerPlantEnableDisable plaEnaDis
          annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
        Modelica.Blocks.Sources.BooleanPulse ySupFan(
          width=60,
          period(displayUnit="h") = 10800,
          startTime(displayUnit="min") = 300)
          annotation (Placement(transformation(extent={{-80,-20},{-60,0}})));
        Modelica.Blocks.Sources.Sine TOut(
          amplitude=10,
          freqHz=1/10800,
          offset=16 + 273.15,
          startTime(displayUnit="min") = 1200)
          annotation (Placement(transformation(extent={{-80,30},{-60,50}})));
        Modelica.Blocks.Sources.IntegerTable yPlaReq(table=[0,0; 800,1; 2500,0; 3000,
              1; 3800,0; 4500,1; 10800,0; 15000,1; 18000,0]) "Plant Request Numbers"
          annotation (Placement(transformation(extent={{-80,-60},{-60,-40}})));
        Modelica.Blocks.Sources.Sine yFanSep(
          amplitude=0.5,
          freqHz=1/10800,
          offset=0.5,
          startTime(displayUnit="min"))
          annotation (Placement(transformation(extent={{-80,-90},{-60,-70}})));
      equation
        connect(ySupFan.y, plaEnaDis.ySupFan) annotation (Line(points={{-59,-10},{-36,
                -10},{-36,0},{-12,0}}, color={255,0,255}));
        connect(TOut.y, plaEnaDis.TOut) annotation (Line(points={{-59,40},{-36,40},{
                -36,4.6},{-12,4.6}},
                             color={0,0,127}));
        connect(yPlaReq.y, plaEnaDis.yPlaReq) annotation (Line(points={{-59,-50},{-34,
                -50},{-34,-4},{-12,-4}}, color={255,127,0}));
        connect(yFanSep.y, plaEnaDis.yFanSpe) annotation (Line(points={{-59,-80},{-32,
                -80},{-32,-7},{-11,-7}}, color={0,0,127}));
        annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          experiment(StopTime=21600, __Dymola_Algorithm="Cvode"));
      end ChillerPlantEnableDisable;

      model PlantRequest
        extends Modelica.Icons.Example;
        Modelica.Blocks.Sources.Sine uPlaReq(
          amplitude=0.5,
          freqHz=1/2000,
          offset=0.5,
          startTime(displayUnit="min") = 300)
          annotation (Placement(transformation(extent={{-60,-10},{-40,10}})));
        FiveZone.Controls.PlantRequest plaReq
          annotation (Placement(transformation(extent={{-8,-10},{12,10}})));
      equation
        connect(uPlaReq.y, plaReq.uPlaVal)
          annotation (Line(points={{-39,0},{-9,0}}, color={0,0,127}));
        annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          experiment(StopTime=21600, __Dymola_Algorithm="Cvode"));
      end PlantRequest;

      model CoolingMode
        "Test the model ChillerWSE.Examples.BaseClasses.CoolingModeController"
        extends Modelica.Icons.Example;

        FiveZone.Controls.CoolingMode cooModCon(
          deaBan1=1,
          deaBan2=1,
          tWai=30,
          deaBan3=1,
          deaBan4=1)
          "Cooling mode controller used in integrared waterside economizer chilled water system"
          annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
        Modelica.Blocks.Sources.Pulse TCHWLeaWSE(
          period=300,
          amplitude=15,
          offset=273.15 + 5) "WSE chilled water supply temperature"
          annotation (Placement(transformation(extent={{-60,-50},{-40,-30}})));
        Modelica.Blocks.Sources.Constant TWetBub(k=273.15 + 5) "Wet bulb temperature"
          annotation (Placement(transformation(extent={{-60,10},{-40,30}})));
        Modelica.Blocks.Sources.Constant TAppTow(k=5) "Cooling tower approach"
          annotation (Placement(transformation(extent={{-60,-20},{-40,0}})));
        Modelica.Blocks.Sources.Constant TCHWEntWSE(k=273.15 + 12)
          "Chilled water return temperature in waterside economizer"
          annotation (Placement(transformation(extent={{-60,-90},{-40,-70}})));
        Modelica.Blocks.Sources.Constant TCHWLeaSet(k=273.15 + 10)
          "Chilled water supply temperature setpoint"
          annotation (Placement(transformation(extent={{-60,40},{-40,60}})));
        Modelica.Blocks.Sources.BooleanPulse yPla(
          width=80,
          period(displayUnit="min") = 300,
          startTime(displayUnit="min") = 60)
          annotation (Placement(transformation(extent={{-60,70},{-40,90}})));
      equation
        connect(TCHWLeaSet.y, cooModCon.TCHWSupSet) annotation (Line(points={{-39,50},
                {-24,50},{-24,5.77778},{-12,5.77778}},
                                           color={0,0,127}));
        connect(TWetBub.y, cooModCon.TWetBul)
          annotation (Line(points={{-39,20},{-26,20},{-26,2.22222},{-12,2.22222}},
                                  color={0,0,127}));
        connect(TAppTow.y, cooModCon.TApp) annotation (Line(points={{-39,-10},{-28,
                -10},{-28,-1.11111},{-12,-1.11111}},
                                  color={0,0,127}));
        connect(TCHWLeaWSE.y, cooModCon.TCHWSupWSE) annotation (Line(points={{-39,-40},
                {-28,-40},{-28,-4.44444},{-12,-4.44444}},
                                              color={0,0,127}));
        connect(TCHWEntWSE.y, cooModCon.TCHWRetWSE) annotation (Line(points={{-39,-80},
                {-26,-80},{-26,-7.77778},{-12,-7.77778}},
                                              color={0,0,127}));
        connect(yPla.y, cooModCon.yPla) annotation (Line(points={{-39,80},{-22,80},{
                -22,8.66667},{-12,8.66667}}, color={255,0,255}));
        annotation (
          Documentation(info="<html>
<p>
This model tests the cooling mode controller implemented in
<a href=\"modelica://FaultInjection.Experimental.SystemLevelFaults.Controls.CoolingMode\">
FaultInjection.Experimental.SystemLevelFaults.Controls.CoolingMode</a>.
</p>
</html>",       revisions="<html>
<ul>
<li>
August 25, 2017, by Yangyang Fu:<br/>
First implementation.
</li>
</ul>
</html>"),
      experiment(
            StartTime=0,
            StopTime=600,
            Tolerance=1e-06),
          __Dymola_Commands(file=
                "Resources/Scripts/dymola/FaultInjection/Experimental/SystemLevelFaults/Controls/Validation/CoolingMode.mos"
              "Simulate and Plot"));
      end CoolingMode;

      model ConstantSpeedPumpStage
        "Test the model ChillerWSE.Examples.BaseClasses.ConstatnSpeedPumpStageControl"
        extends Modelica.Icons.Example;

        FiveZone.Controls.ConstantSpeedPumpStage conSpePumSta(tWai=30)
          "Staging controller for constant speed pumps"
          annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
        Modelica.Blocks.Sources.IntegerTable cooMod(table=[360,1; 720,2; 1080,3; 1440,
              4])
          "Cooling mode"
          annotation (Placement(transformation(extent={{-60,40},{-40,60}})));
        Modelica.Blocks.Sources.IntegerTable chiNumOn(
          table=[0,0; 360,1; 540,2; 720,1;
                 900,2; 1080,1; 1260,2; 1440,1])
          "The number of running chillers"
          annotation (Placement(transformation(extent={{-60,-40},{-40,-20}})));
      equation
        connect(cooMod.y, conSpePumSta.cooMod)
          annotation (Line(points={{-39,50},{-20,50},{-20,5},{-12,5}},
                              color={255,127,0}));
        connect(chiNumOn.y,conSpePumSta.numOnChi)
          annotation (Line(points={{-39,-30},{-20,-30},{-20,-5},{-12,-5}},
                              color={255,127,0}));
        annotation (    __Dymola_Commands(file=
                "modelica://Buildings/Resources/Scripts/Dymola/Applications/DataCenters/ChillerCooled/Controls/Validation/ConstantSpeedPumpStage.mos"
              "Simulate and plot"),
          Documentation(info="<html>
<p>
This example test how the number of required constant-speed pumps varies
based on cooling mode signals and the number of running chillers. Detailed
control logic can be found in
<a href=\"modelica://Buildings.Applications.DataCenters.ChillerCooled.Controls.ConstantSpeedPumpStage\">
Buildings.Applications.DataCenters.ChillerCooled.Controls.ConstantSpeedPumpStage</a>.
</p>
</html>",       revisions="<html>
<ul>
<li>
August 25, 2017, by Yangyang Fu:<br/>
First implementation.
</li>
</ul>
</html>"),
      experiment(
            StartTime=0,
            StopTime=1440,
            Tolerance=1e-06));
      end ConstantSpeedPumpStage;

      model CoolingTowerSpeed
        "Test the model ChillerWSE.Examples.BaseClasses.CoolingTowerSpeedControl"
        extends Modelica.Icons.Example;

        parameter Modelica.Blocks.Types.SimpleController controllerType=
          Modelica.Blocks.Types.SimpleController.PID
          "Type of controller"
          annotation(Dialog(tab="Controller"));
        parameter Real k(min=0, unit="1") = 1
          "Gain of controller"
          annotation(Dialog(tab="Controller"));
        parameter Modelica.SIunits.Time Ti(min=Modelica.Constants.small)=0.5
          "Time constant of integrator block"
           annotation (Dialog(enable=
                (controllerType == Modelica.Blocks.Types.SimpleController.PI or
                controllerType == Modelica.Blocks.Types.SimpleController.PID),tab="Controller"));
        parameter Modelica.SIunits.Time Td(min=0)=0.1
          "Time constant of derivative block"
           annotation (Dialog(enable=
                (controllerType == Modelica.Blocks.Types.SimpleController.PD or
                controllerType == Modelica.Blocks.Types.SimpleController.PID),tab="Controller"));
        parameter Real yMax(start=1)=1
         "Upper limit of output"
          annotation(Dialog(tab="Controller"));
        parameter Real yMin=0
         "Lower limit of output"
          annotation(Dialog(tab="Controller"));

        FiveZone.Controls.CoolingTowerSpeed cooTowSpeCon(controllerType=
              Modelica.Blocks.Types.SimpleController.PI)
          "Cooling tower speed controller"
          annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
        Modelica.Blocks.Sources.Sine CHWST(
          amplitude=2,
          freqHz=1/360,
          offset=273.15 + 5)
          "Chilled water supply temperature"
          annotation (Placement(transformation(extent={{-60,-80},{-40,-60}})));
        Modelica.Blocks.Sources.Constant CWSTSet(k=273.15 + 20)
          "Condenser water supply temperature setpoint"
          annotation (Placement(transformation(extent={{-60,70},{-40,90}})));
        Modelica.Blocks.Sources.Sine CWST(
          amplitude=5,
          freqHz=1/360,
          offset=273.15 + 20)
          "Condenser water supply temperature"
          annotation (Placement(transformation(extent={{-60,-40},{-40,-20}})));
        Modelica.Blocks.Sources.Constant CHWSTSet(k=273.15 + 6)
          "Chilled water supply temperature setpoint"
          annotation (Placement(transformation(extent={{-60,0},{-40,20}})));
        Modelica.Blocks.Sources.IntegerTable cooMod(table=[360,1; 720,2; 1080,3; 1440,
              4])
          "Cooling mode"
          annotation (Placement(transformation(extent={{-60,40},{-40,60}})));
      equation
        connect(CWSTSet.y, cooTowSpeCon.TCWSupSet)
          annotation (Line(points={{-39,80},{-20,80},{-20,80},{-20,22},{-20,10},{-12,
                10}},                                         color={0,0,127}));
        connect(CHWSTSet.y, cooTowSpeCon.TCHWSupSet)
          annotation (Line(points={{-39,10},
                {-32,10},{-32,1.11111},{-12,1.11111}}, color={0,0,127}));
        connect(CWST.y, cooTowSpeCon.TCWSup)
          annotation (Line(points={{-39,-30},{-32,-30},
                {-32,-3.33333},{-12,-3.33333}}, color={0,0,127}));
        connect(CHWST.y, cooTowSpeCon.TCHWSup)
          annotation (Line(points={{-39,-70},{-32,
                -70},{-24,-70},{-24,-7.77778},{-12,-7.77778}}, color={0,0,127}));
        connect(cooMod.y, cooTowSpeCon.cooMod)
          annotation (Line(points={{-39,50},{-26,50},{-26,5.55556},{-12,5.55556}},
                                                  color={255,127,0}));
        annotation (    __Dymola_Commands(file=
              "modelica://Buildings/Resources/Scripts/Dymola/Applications/DataCenters/ChillerCooled/Controls/Validation/CoolingTowerSpeed.mos"
              "Simulate and plot"),
          Documentation(info="<html>
<p>
This example tests the controller for the cooling tower fan speed. Detailed control logic can be found in
<a href=\"modelica://Buildings.Applications.DataCenters.ChillerCooled.Controls.CoolingTowerSpeed\">
Buildings.Applications.DataCenters.ChillerCooled.Controls.CoolingTowerSpeed</a>.
</p>
</html>",       revisions="<html>
<ul>
<li>
August 25, 2017, by Yangyang Fu:<br/>
First implementation.
</li>
</ul>
</html>"),
      experiment(
            StopTime=2000,
            Tolerance=1e-06,
            __Dymola_Algorithm="Dassl"));
      end CoolingTowerSpeed;

      model MinimumFlowBypassValve
        extends Modelica.Icons.Example;
        Modelica.Blocks.Sources.Sine m_flow(
          amplitude=0.05,
          freqHz=1/10000,
          offset=0.1,
          startTime(displayUnit="min") = 60)
          annotation (Placement(transformation(extent={{-60,10},{-40,30}})));
        FiveZone.Controls.MinimumFlowBypassValve minFloBypVal(m_flow_minimum=
              0.13, controllerType=Modelica.Blocks.Types.SimpleController.PI)
          annotation (Placement(transformation(extent={{-12,-10},{8,10}})));
        Modelica.Blocks.Sources.BooleanConstant boo
          annotation (Placement(transformation(extent={{-60,-40},{-40,-20}})));
      equation
        connect(m_flow.y, minFloBypVal.m_flow) annotation (Line(points={{-39,20},{-25.5,
                20},{-25.5,3},{-14,3}}, color={0,0,127}));
        connect(boo.y, minFloBypVal.yPla) annotation (Line(points={{-39,-30},{-26,-30},
                {-26,-3},{-14,-3}}, color={255,0,255}));
        annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          experiment(StopTime=21600, __Dymola_Algorithm="Cvode"));
      end MinimumFlowBypassValve;

      model HotWaterTemperatureReset
        extends Modelica.Icons.Example;
        Modelica.Blocks.Sources.Sine yVal(
          amplitude=0.3,
          freqHz=1/8000,
          offset=0.7,
          startTime(displayUnit="min") = 0)
          annotation (Placement(transformation(extent={{-60,-30},{-40,-10}})));
        FiveZone.Controls.HotWaterTemperatureReset hotWatTemRes(resAmo=2)
          annotation (Placement(transformation(extent={{-10,-8},{10,12}})));
        Modelica.Blocks.Sources.BooleanPulse yPla(
          width=80,
          period(displayUnit="min") = 12000,
          startTime(displayUnit="min") = 600)
          annotation (Placement(transformation(extent={{-60,20},{-40,40}})));
      equation
        connect(yPla.y, hotWatTemRes.uDevSta) annotation (Line(points={{-39,30},{-26,
                30},{-26,9.4},{-12,9.4}}, color={255,0,255}));
        connect(yVal.y, hotWatTemRes.uPlaHeaVal) annotation (Line(points={{-39,-20},{
                -26,-20},{-26,2},{-12,2}}, color={0,0,127}));
        annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          experiment(StopTime=21600, __Dymola_Algorithm="Cvode"),
          __Dymola_Commands(file="\"\"" "Simulate and Plot"));
      end HotWaterTemperatureReset;
    annotation (Icon(graphics={
            Rectangle(
              lineColor={200,200,200},
              fillColor={248,248,248},
              fillPattern=FillPattern.HorizontalCylinder,
              extent={{-100,-100},{100,100}},
              radius=25.0),
            Polygon(
              origin={8,14},
              lineColor={78,138,73},
              fillColor={78,138,73},
              pattern=LinePattern.None,
              fillPattern=FillPattern.Solid,
              points={{-58.0,46.0},{42.0,-14.0},{-58.0,-74.0},{-58.0,46.0}}),
            Rectangle(
              lineColor={128,128,128},
              extent={{-100,-100},{100,100}},
              radius=25.0)}));
    end Validation;

    package BaseClasses

      model TimeLessEqual
        "Timer calculating the time when A is less than or equal than B"

        parameter Real threshold=0 "Comparison with respect to threshold";

        Modelica.Blocks.Interfaces.RealOutput y(
          final quantity="Time",
          final unit="s")
          "Connector of Real output signal"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));

        Modelica.Blocks.Logical.LessEqualThreshold    lesEqu(
           threshold = threshold)
          annotation (Placement(transformation(extent={{-30,-10},{-10,10}})));

        Modelica.Blocks.Logical.Timer tim "Timer"
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));

        Modelica.Blocks.Interfaces.IntegerInput u1
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Math.IntegerToReal intToRea
          annotation (Placement(transformation(extent={{-80,-10},{-60,10}})));
      equation
        connect(lesEqu.y, tim.u)
          annotation (Line(points={{-9,0},{18,0}},  color={255,0,255}));
        connect(tim.y,y)  annotation (Line(points={{41,0},{110,0}}, color={0,0,127}));
        connect(intToRea.y, lesEqu.u)
          annotation (Line(points={{-59,0},{-32,0}}, color={0,0,127}));
        connect(u1, intToRea.u)
          annotation (Line(points={{-120,0},{-82,0}}, color={255,127,0}));
        annotation (defaultComponentName="lesEqu",
        Icon(coordinateSystem(preserveAspectRatio=false), graphics={
              Rectangle(
                extent={{-100,100},{100,-100}},
                lineColor={0,0,0},
                lineThickness=5.0,
                fillColor={210,210,210},
                fillPattern=FillPattern.Solid,
                borderPattern=BorderPattern.Raised),
                                         Text(
                extent={{-90,-40},{60,40}},
                lineColor={0,0,0},
                textString="<="),
              Ellipse(
                extent={{71,7},{85,-7}},
                lineColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillPattern=FillPattern.Solid),
                                              Text(
              extent={{-150,150},{150,110}},
              textString="%name",
              lineColor={0,0,255})}),                                  Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          Documentation(info="<html>
<p>This model represents a timer that starts to calculate the time when the input is less than or equal to a certain threshold. It will return to zero when the condition does not satisfy.</p>
</html>"));
      end TimeLessEqual;

      model TimeLessEqualRea
        "Timer calculating the time when A is less than or equal than B"

        parameter Real threshold=0 "Comparison with respect to threshold";

        Modelica.Blocks.Interfaces.RealOutput y(
          final quantity="Time",
          final unit="s")
          "Connector of Real output signal"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));

        Modelica.Blocks.Logical.LessEqualThreshold    lesEqu(
           threshold = threshold)
          annotation (Placement(transformation(extent={{-30,-10},{-10,10}})));

        Modelica.Blocks.Logical.Timer tim "Timer"
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));

        Modelica.Blocks.Interfaces.RealInput u1
                                      "Connector of Real input signal"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
      equation
        connect(lesEqu.y, tim.u)
          annotation (Line(points={{-9,0},{18,0}},  color={255,0,255}));
        connect(tim.y,y)  annotation (Line(points={{41,0},{110,0}}, color={0,0,127}));
        connect(lesEqu.u, u1)
          annotation (Line(points={{-32,0},{-120,0}}, color={0,0,127}));
        annotation (defaultComponentName="lesEqu",
        Icon(coordinateSystem(preserveAspectRatio=false), graphics={
              Rectangle(
                extent={{-100,100},{100,-100}},
                lineColor={0,0,0},
                lineThickness=5.0,
                fillColor={210,210,210},
                fillPattern=FillPattern.Solid,
                borderPattern=BorderPattern.Raised),
                                         Text(
                extent={{-90,-40},{60,40}},
                lineColor={0,0,0},
                textString="<="),
              Ellipse(
                extent={{71,7},{85,-7}},
                lineColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillPattern=FillPattern.Solid),
                                              Text(
              extent={{-150,150},{150,110}},
              textString="%name",
              lineColor={0,0,255})}),                                  Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          Documentation(info="<html>
<p>This model represents a timer that starts to calculate the time when the input is less than or equal to a certain threshold. It will return to zero when the condition does not satisfy.</p>
</html>"));
      end TimeLessEqualRea;
    end BaseClasses;
  annotation (Documentation(info="<html>
<p>Collection of models for the control of airside and waterside systems. </p>
</html>"),   Icon(graphics={
        Rectangle(
          origin={10,45.1488},
          fillColor={255,255,255},
          extent={{-30.0,-20.1488},{30.0,20.1488}}),
        Polygon(
          origin={-30,45},
          pattern=LinePattern.None,
          fillPattern=FillPattern.Solid,
          points={{10.0,0.0},{-5.0,5.0},{-5.0,-5.0}}),
        Line(
          origin={-41.25,10},
          points={{21.25,-35.0},{-13.75,-35.0},{-13.75,35.0},{6.25,35.0}}),
        Rectangle(
          origin={10,-24.8512},
          fillColor={255,255,255},
          extent={{-30.0,-20.1488},{30.0,20.1488}}),
        Line(
          origin={61.25,10},
          points={{-21.25,35.0},{13.75,35.0},{13.75,-35.0},{-6.25,-35.0}})}));
  end Controls;

  package PrimarySideControl "Package with primary chilled water loop control"
    extends Modelica.Icons.Package;

    package CHWLoopEquipment "Collection of local controls in the chilled water loop"
    extends Modelica.Icons.Package;

      model StageLoadBasedChiller
        "Chiller staging control based on cooling load"
        extends Modelica.Blocks.Icons.Block;
        parameter Modelica.SIunits.Power QEva_nominal
          "Nominal cooling capaciaty(Negative means cooling)";
        parameter Integer numChi=2 "Design number of chillers";
        parameter Real staUpThr=0.8 "Staging up threshold";
        parameter Real staDowThr=0.25 "Staging down threshold";
        parameter Modelica.SIunits.Time waiTimStaUp=300
          "Time duration of for staging up";
        parameter Modelica.SIunits.Time waiTimStaDow=300
          "Time duration of for staging down";
        parameter Modelica.SIunits.Time shoCycTim=1200
          "Time duration to avoid short cycling of equipment";
        inner Modelica.StateGraph.StateGraphRoot stateGraphRoot
          annotation (Placement(transformation(extent={{-100,80},{-80,100}})));

        Modelica.Blocks.Interfaces.RealInput QTot(unit="W")
          "Total cooling load in the chillers, negative"
          annotation (Placement(transformation(extent={{-140,-60},{-100,-20}}),
              iconTransformation(extent={{-140,-60},{-100,-20}})));

        Modelica.Blocks.Sources.BooleanExpression unOccFre(y=uOpeMod == Integer(FiveZone.Types.CoolingModes.Off)
               or uOpeMod == Integer(FiveZone.Types.CoolingModes.FreeCooling))
          "Unoccupied or FreeCooling mode"
          annotation (Placement(transformation(extent={{-92,10},{-72,30}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi1
          annotation (Placement(transformation(extent={{-20,10},{0,30}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant zer(k=0) "Zero"
          annotation (Placement(transformation(extent={{-62,40},{-42,60}})));
        Buildings.Controls.OBC.CDL.Conversions.RealToInteger reaToInt
          annotation (Placement(transformation(extent={{10,10},{30,30}})));
        BaseClasses.SequenceSignal seqSig(n=numChi)
          "Simple model that is used to determine the on and off sequence of equipment"
          annotation (Placement(transformation(extent={{50,10},{70,30}})));
        BaseClasses.Stage sta(
          shoCycTim=shoCycTim,
          waiTimStaUp=waiTimStaUp,
          waiTimStaDow=waiTimStaDow,
          staUpThr=staUpThr*(-QEva_nominal),
          staDowThr=staDowThr*(-QEva_nominal))
          annotation (Placement(transformation(extent={{-20,-54},{0,-34}})));
        Buildings.Controls.OBC.CDL.Conversions.IntegerToReal intToRea1
          annotation (Placement(transformation(extent={{12,-54},{32,-34}})));
        Modelica.Blocks.Interfaces.RealOutput y[numChi]
          "On and off signal of chiller"
          annotation (Placement(transformation(extent={{100,-50},{120,-30}})));
        Buildings.Controls.OBC.CDL.Interfaces.IntegerOutput yChi
          "Number of active chillers"
          annotation (Placement(transformation(extent={{100,30},{120,50}}),
              iconTransformation(extent={{100,30},{120,50}})));
        Modelica.Blocks.Interfaces.IntegerInput uOpeMod
          "Cooling mode in WSEControls.Type.OperationaModes" annotation (Placement(
              transformation(extent={{-140,20},{-100,60}}),  iconTransformation(
                extent={{-140,20},{-100,60}})));

        Modelica.Blocks.Math.Gain gain(k=-1)
          annotation (Placement(transformation(extent={{-80,-50},{-60,-30}})));
        Buildings.Controls.OBC.CDL.Logical.Not not2
          annotation (Placement(transformation(extent={{-60,-20},{-40,0}})));
      equation
        connect(zer.y,swi1. u1) annotation (Line(points={{-40,50},{-30,50},{-30,28},{-22,
                28}},    color={0,0,127}));
        connect(unOccFre.y, swi1.u2)
          annotation (Line(points={{-71,20},{-22,20}}, color={255,0,255}));
        connect(swi1.y,reaToInt. u)
          annotation (Line(points={{2,20},{8,20}}, color={0,0,127}));
        connect(reaToInt.y,seqSig. u)
          annotation (Line(points={{32,20},{48,20}}, color={255,127,0}));
        connect(reaToInt.y,yChi)  annotation (Line(points={{32,20},{40,20},{40,40},{110,
                40}}, color={255,127,0}));
        connect(seqSig.y,y)  annotation (Line(points={{71,20},{80,20},{80,-40},{110,-40}},
              color={0,0,127}));
        connect(sta.ySta,intToRea1. u)
          annotation (Line(points={{1,-44},{10,-44}},    color={255,127,0}));
        connect(gain.y, sta.u) annotation (Line(points={{-59,-40},{-22,-40}},
                                      color={0,0,127}));
        connect(QTot, gain.u) annotation (Line(points={{-120,-40},{-82,-40}},
              color={0,0,127}));
        connect(unOccFre.y, not2.u) annotation (Line(points={{-71,20},{-66,20},{-66,
                -10},{-62,-10}}, color={255,0,255}));
        connect(not2.y, sta.on) annotation (Line(points={{-38,-10},{-32,-10},{-32,-48},
                {-22,-48}}, color={255,0,255}));
        connect(intToRea1.y, swi1.u3) annotation (Line(points={{34,-44},{40,-44},{40,0},
                {-30,0},{-30,12},{-22,12}},    color={0,0,127}));
       annotation (
          defaultComponentName="staLoaChi",
          Documentation(info="<html>

<p>This model describes a chiller staging control based on the part load ratio (PLR) or cooling load Q.</p>
<ul>
<li>
In unoccupied and free cooling mode, the chillers are off.
</li>

<li>
In pre-partial, partial and full mechanical cooling mode, the chillers are staged based on part load ratio or cooling load in chillers. At the beginning, the number of chillers stay unchanged
from previous operating mode.
</li>

</ul>

<h4>PLR or Q-based Stage Control </h4>

<p>Chillers are staged up when</p>
<ol>
<li>Current stage has been activated for at least <i>30</i> minutes (<i><span style=\"font-size: 10pt;\">△</span>t<sub>stage,on</sub> &gt; 30 min) </i>and</li>
<li>PLR for any active chiller is greater than <i>80</i>&percnt; for <i>10</i> minutes <i>(PLR<sub>chiller</sub> &gt; 80&percnt; for 10 min).</i></li>
</ol>
<p>Chillers are staged down when</p>
<ol>
<li>Current stage has been activated for at least <i>30</i> minutes <i>(<span style=\"font-size: 10pt;\">△</span>t<sub>stage,on</sub> &gt; 30 min)</i> and</li>
<li>PLR for any active chiller is less than 25&percnt; for 15 minutes <i>(PLR<sub>chiller</sub> &lt; 25&percnt; for 15 min)</i>.</li>
</ol>
<p>It is noted that the time duration and the percentage can be adjusted according to different projects.</p>
<p>This control logic is provided by Jeff Stein via email communication.</p>
</html>",       revisions="<html>
<ul>
<li>August 16, 2018, by Yangyang Fu:<br> 
Improve documentation. 
</li>
<li>June 12, 2018, by Xing Lu:<br>
First implementation. 
</li>
</ul>
</html>"),Diagram(coordinateSystem(extent={{-100,-100},{100,100}})),
          Icon(coordinateSystem(extent={{-100,-100},{100,100}})),
          __Dymola_Commands);
      end StageLoadBasedChiller;

      model StagePump "Staging control for CHW pumps"
        extends Modelica.Blocks.Icons.Block;
        parameter Modelica.SIunits.MassFlowRate m_flow_nominal
          "Nominal mass flow rate of the CHW pump";
        parameter Integer numPum=2 "Design number of pumps";
        parameter Real staUpThr = 0.85 "Staging up threshold";
        parameter Real staDowThr = 0.45 "Staging down threshold";
        parameter Modelica.SIunits.Time waiTimStaUp=300
          "Time duration of for staging up";
        parameter Modelica.SIunits.Time waiTimStaDow=300
          "Time duration of for staging down";
        parameter Modelica.SIunits.Time shoCycTim=1200
          "Time duration to avoid short cycling of equipment";
        Modelica.Blocks.Interfaces.RealInput masFloPum
          "Average mass flowrate of the active CHW pump"
          annotation (Placement(transformation(extent={{-140,
                  -60},{-100,-20}}),
              iconTransformation(extent={{-140,-60},{-100,
                  -20}})));

        inner Modelica.StateGraph.StateGraphRoot stateGraphRoot
          annotation (Placement(transformation(extent={{-100,80},{-80,100}})));

        Modelica.Blocks.Sources.BooleanExpression unOcc(y=uOpeMod
               == Integer(FiveZone.Types.CoolingModes.Off))
          "Unoccupied or FreeCooling mode"
          annotation (Placement(transformation(extent={{-90,10},{-70,30}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi1
          annotation (Placement(transformation(extent={{-20,10},{0,30}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant zer(k=0) "Zero"
          annotation (Placement(transformation(extent={{-60,40},{-40,60}})));
        Buildings.Controls.OBC.CDL.Conversions.RealToInteger reaToInt
          annotation (Placement(transformation(extent={{10,10},{30,30}})));
        BaseClasses.SequenceSignal seqSig(n=numPum)
          "Simple model that is used to determine the on and off sequence of equipment"
          annotation (Placement(transformation(extent={{50,10},{70,30}})));
        BaseClasses.Stage sta(
          shoCycTim=shoCycTim,
          waiTimStaUp=waiTimStaUp,
          waiTimStaDow=waiTimStaDow,
          staUpThr=staUpThr*m_flow_nominal,
          staDowThr=staDowThr*m_flow_nominal)
          annotation (Placement(transformation(extent={{-20,-54},{0,-34}})));
        Buildings.Controls.OBC.CDL.Conversions.IntegerToReal intToRea1
          annotation (Placement(transformation(extent={{12,-54},{32,-34}})));
        Modelica.Blocks.Interfaces.RealOutput y[numPum]
          "On and off signal of pumps"
          annotation (Placement(transformation(extent={{100,-50},{120,-30}})));
        Buildings.Controls.OBC.CDL.Interfaces.IntegerOutput yPum
          "Number of active pumps"
          annotation (Placement(transformation(extent={{100,30},{120,50}}),
              iconTransformation(extent={{100,30},{120,50}})));
        Modelica.Blocks.Interfaces.IntegerInput uOpeMod
          "Cooling mode in WSEControls.Type.OperationaModes" annotation (Placement(
              transformation(extent={{-140,20},{-100,60}}),  iconTransformation(
                extent={{-140,20},{-100,60}})));
        Buildings.Controls.OBC.CDL.Logical.Not not2
          annotation (Placement(transformation(extent={{-60,-30},{-40,-10}})));
      equation
        connect(zer.y,swi1. u1) annotation (Line(points={{-38,50},{-30,50},{-30,28},{-22,
                28}},    color={0,0,127}));
        connect(unOcc.y,swi1. u2)
          annotation (Line(points={{-69,20},{-22,20}}, color={255,0,255}));
        connect(swi1.y,reaToInt. u)
          annotation (Line(points={{2,20},{8,20}}, color={0,0,127}));
        connect(reaToInt.y,seqSig. u)
          annotation (Line(points={{32,20},{48,20}}, color={255,127,0}));
        connect(reaToInt.y,yPum)  annotation (Line(points={{32,20},{40,20},{40,40},{110,
                40}}, color={255,127,0}));
        connect(seqSig.y,y)  annotation (Line(points={{71,20},{80,20},{80,-40},{110,-40}},
              color={0,0,127}));
        connect(sta.ySta,intToRea1. u)
          annotation (Line(points={{1,-44},{10,-44}},    color={255,127,0}));
        connect(masFloPum, sta.u) annotation (Line(
              points={{-120,-40},{-22,-40}},
                       color={0,0,127}));
        connect(unOcc.y, not2.u) annotation (Line(points={{-69,20},{-66,20},{-66,-20},
                {-62,-20}}, color={255,0,255}));
        connect(not2.y, sta.on) annotation (Line(points={{-38,-20},{-30,-20},{-30,-48},
                {-22,-48}}, color={255,0,255}));
        connect(intToRea1.y, swi1.u3) annotation (Line(points={{34,-44},{40,-44},{40,0},
                {-30,0},{-30,12},{-22,12}},    color={0,0,127}));
        annotation (defaultComponentName="staPum",
        Documentation(info="<html>
<p>This model describes a chilled water pump staging control. </p>
<ul>
<li>In unoccupied and free cooling mode, the chillers are off. </li>
<li>In pre-partial, partial and full mechanical cooling mode, the chilled water pumps are staged based on measured flowrate. At the beginning, the number of pumps stay unchanged from previous operating mode. </li>
</ul>
<h4>Flowrate-based Stage Control </h4>
<p>The CHW pumps are staged up when </p>
<ol>
<li>Current stage has been active for at least 15 minutes (<i><span style=\"font-size: 10pt;\">△</span>t<sub>stage,on</sub> &gt; 15 min) </i>and </li>
<li>The measured flowrate is larger than 85&percnt; of the total nominal flowrate of the active pumps for 2 minutes <i>(m<sub>CHWP</sub> &gt; 85&percnt; &middot; m<sub>CHWP,nominal</sub> for 2 min)</i>.</li>
</ol>
<p>The CHW pumps are staged down when</p>
<ol>
<li>Current stage has been active for at least 15 minutes (<i><span style=\"font-size: 10pt;\">△</span>t<sub>stage,on</sub> &gt; 15 min) </i>and</li>
<li>The measured flowrate is less than 45&percnt; of the total nominal flowrate of the active pumps for 15 minutes <i>(m<sub>CHWP</sub> &lt; 45&percnt; &middot; m<sub>CHWP,nominal </sub>for 15 min)</i>.</li>
</ol>
<p>This control logic is provided by Jeff Stein via email communication.</p>
</html>",       revisions="<html>
<ul>
<li>June 14, 2018, by Xing Lu:<br>First implementation. </li>
</ul>
</html>"),Diagram(coordinateSystem(extent={{-100,-100},{100,100}})),
          Icon(coordinateSystem(extent={{-100,-100},{100,100}})),
          __Dymola_Commands);
      end StagePump;

      model TemperatureDifferentialPressureReset
        "CHWST and CHW DP reset control for chillers"
        extends Modelica.Blocks.Icons.Block;
        parameter Modelica.SIunits.Time samplePeriod=120 "Sample period of component";
        parameter Real uTri=0 "Value to triggering the request for actuator";
        parameter Real yEqu0=0 "y setpoint when equipment starts";
        parameter Real yDec=-0.03 "y decrement (must be negative)";
        parameter Real yInc=0.03 "y increment (must be positive)";
        parameter Real x1=0.5 "First interval [x0, x1] and second interval (x1, x2]"
        annotation(Dialog(tab="Pressure and temperature reset points"));
        parameter Modelica.SIunits.Pressure dpMin = 100 "dpmin"
        annotation(Dialog(tab="Pressure and temperature reset points"));
        parameter Modelica.SIunits.Pressure dpMax =  300 "dpmax"
        annotation(Dialog(tab="Pressure and temperature reset points"));
        parameter Modelica.SIunits.ThermodynamicTemperature TMin=273.15+5.56 "Tchi,min"
        annotation(Dialog(tab="Pressure and temperature reset points"));
        parameter Modelica.SIunits.ThermodynamicTemperature TMax = 273.15+22 "Tchi,max"
        annotation(Dialog(tab="Pressure and temperature reset points"));
        parameter Modelica.SIunits.Time startTime=0 "First sample time instant";

        Modelica.Blocks.Interfaces.RealInput u
          "Input signall, such as dT, or valve position"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
        BaseClasses.LinearPiecewiseTwo linPieTwo(
          x0=0,
          x1=x1,
          x2=1,
          y10=dpMin,
          y11=dpMax,
          y20=TMax,
          y21=TMin)
          "Calculation of two piecewise linear functions"
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));

        Modelica.Blocks.Interfaces.RealOutput dpSet(
          final quantity="Pressure",
          final unit = "Pa") "DP setpoint"
          annotation (Placement(transformation(extent={{100,40},{120,60}}),
              iconTransformation(extent={{100,40},{120,60}})));
        Modelica.Blocks.Interfaces.RealOutput TSet(
          final quantity="ThermodynamicTemperature",
          final unit="K")
          "CHWST"
          annotation (Placement(transformation(extent={{100,-60},{120,-40}}),
              iconTransformation(extent={{100,-60},{120,-40}})));
        BaseClasses.TrimAndRespond triAndRes(
          samplePeriod=samplePeriod,
          startTime=startTime,
          uTri=uTri,
          yEqu0=yEqu0,
          yDec=yDec,
          yInc=yInc)
          annotation (Placement(transformation(extent={{-40,-10},{-20,10}})));

        Modelica.Blocks.Logical.Switch swi1 "Switch"
          annotation (Placement(transformation(extent={{60,70},{80,90}})));
        Modelica.Blocks.Sources.RealExpression dpSetIni(y=dpMin)
          "Initial dp setpoint"
          annotation (Placement(transformation(extent={{-20,80},{0,100}})));
        Modelica.Blocks.Sources.RealExpression TSetIni(y=TMin)
          "Initial temperature setpoint"
          annotation (Placement(transformation(extent={{-20,-88},{0,-68}})));
        Modelica.Blocks.Logical.Switch swi2 "Switch"
          annotation (Placement(transformation(extent={{60,-80},{80,-60}})));
        Modelica.Blocks.Interfaces.IntegerInput uOpeMod
          "Cooling mode in WSEControls.Type.OperationaModes" annotation (Placement(
              transformation(extent={{-140,40},{-100,80}}),  iconTransformation(
                extent={{-140,40},{-100,80}})));
        Buildings.Controls.OBC.CDL.Integers.GreaterThreshold intGreThr(threshold=
              Integer(Buildings.Applications.DataCenters.Types.CoolingModes.FreeCooling))
          annotation (Placement(transformation(extent={{-60,50},{-40,70}})));
      equation

        connect(triAndRes.y, linPieTwo.u)
          annotation (Line(points={{-19,0},{18,0}}, color={0,0,127}));
        connect(uOpeMod, intGreThr.u)
          annotation (Line(points={{-120,60},{-62,60}}, color={255,127,0}));
        connect(intGreThr.y, swi1.u2) annotation (Line(points={{-38,60},{50,60},{50,
                80},{58,80}}, color={255,0,255}));
        connect(intGreThr.y, swi2.u2) annotation (Line(points={{-38,60},{50,60},{50,
                -70},{58,-70}}, color={255,0,255}));
        connect(linPieTwo.y[1], swi1.u1) annotation (Line(points={{41,-0.5},{48,-0.5},
                {48,88},{58,88}}, color={0,0,127}));
        connect(dpSetIni.y, swi1.u3) annotation (Line(points={{1,90},{46,90},{46,72},
                {58,72}}, color={0,0,127}));
        connect(linPieTwo.y[2], swi2.u1) annotation (Line(points={{41,0.5},{52,0.5},{
                52,-62},{58,-62}}, color={0,0,127}));
        connect(TSetIni.y, swi2.u3)
          annotation (Line(points={{1,-78},{58,-78}}, color={0,0,127}));
        connect(swi1.y, dpSet) annotation (Line(points={{81,80},{90,80},{90,50},{110,
                50}}, color={0,0,127}));
        connect(swi2.y, TSet) annotation (Line(points={{81,-70},{90,-70},{90,-50},{
                110,-50}}, color={0,0,127}));
        connect(u, triAndRes.u)
          annotation (Line(points={{-120,0},{-42,0}}, color={0,0,127}));
        annotation (defaultComponentName="temDifPreRes",
          Documentation(info="<html>
<p>This model describes a chilled water supply temperature setpoint and differential pressure setpoint reset control. In this logic, it is to first increase the different pressure, <i>&Delta;p</i>, of the chilled water loop to increase the mass flow rate. If <i>&Delta;p</i> reaches the maximum value and further cooling is still needed, the chiller remperature setpoint, <i>T<sub>chi,set</i></sub>, is reduced. If there is too much cooling, the <i>T<sub>chi,set</i></sub> and <i>&Delta;p</i> will be changed in the reverse direction. </p>
<p>The model implements a discrete time trim and respond logic as follows: </p>
<ul>
<li>A cooling request is triggered if the input signal, <i>y</i>, is larger than 0. <i>y</i> is the difference between the actual and set temperature of the suppuly air to the data center room.</li>
<li>The request is sampled every 2 minutes. If there is a cooling request, the control signal <i>u</i> is increased by <i>0.03</i>, where <i>0 &le; u &le; 1</i>. If there is no cooling request, <i>u</i> is decreased by <i>0.03</i>. </li>
</ul>
<p>The control signal <i>u</i> is converted to setpoints for <i>&Delta;p</i> and <i>T<sub>chi,set</i></sub> as follows: </p>
<ul>
<li>If <i>u &isin; [0, x]</i> then <i>&Delta;p = &Delta;p<sub>min</sub> + u &nbsp;(&Delta;p<sub>max</sub>-&Delta;p<sub>min</sub>)/x</i> and <i>T = T<sub>max</i></sub></li>
<li>If <i>u &isin; (x, 1]</i> then <i>&Delta;p = &Delta;p<sub>max</i></sub> and <i>T = T<sub>max</sub> - (u-x)&nbsp;(T<sub>max</sub>-T<sub>min</sub>)/(1-x) </i></li>
</ul>
<p>where <i>&Delta;p<sub>min</i></sub> and <i>&Delta;p<sub>max</i></sub> are minimum and maximum values for <i>&Delta;p</i>, and <i>T<sub>min</i></sub> and <i>T<sub>max</i></sub> are the minimum and maximum values for <i>T<sub>chi,set</i></sub>. </p>
<p>Note that we deactivate the trim and response when the chillers are off.</p>

<h4>Reference</h4>
<p>Stein, J. (2009). Waterside Economizing in Data Centers: Design and Control Considerations. ASHRAE Transactions, 115(2), 192-200.</p>
<p>Taylor, S.T. (2007). Increasing Efficiency with VAV System Static Pressure Setpoint Reset. ASHRAE Journal, June, 24-32. </p>
</html>",       revisions="<html>
<ul>
<li><i>December 19, 2018</i> by Yangyang Fu:<br/>
        Deactivate reset when chillers are off.
</li>
<li><i>June 23, 2018</i> by Xing Lu:<br/>
        First implementation.
</li>
</ul>
</html>"));
      end TemperatureDifferentialPressureReset;

      annotation (Documentation(info="<html>
<p>This package contains a collection of the local controls in the chilled water loop.</p>
</html>"));
    end CHWLoopEquipment;

    package CWLoopEquipment "Collection of local controls in the condenser water loop"
    extends Modelica.Icons.Package;

      model MaximumSpeedFan
        "The maximum fan speed in cooling towers are reset based on the operation mode"
          extends Modelica.Blocks.Icons.Block;
        parameter Real lowMax = 0.9 "Low value of maximum speed";
        parameter Real pmcMax = 0.95 "Maximum speed in PMC mode";
        parameter Integer numPum = 2 "Number of design pumps in condenser water loop";

        Modelica.Blocks.Interfaces.IntegerInput uOpeMod
          "Cooling mode in WSEControls.Type.OperationaModes" annotation (Placement(
              transformation(extent={{-140,-60},{-100,-20}}),iconTransformation(
                extent={{-140,-60},{-100,-20}})));
        Modelica.Blocks.Interfaces.IntegerInput
                                             numActPum
          "Number of active pumps in condenser water loop"
          annotation (Placement(transformation(extent={{-140,20},{-100,60}})));
        Buildings.Controls.OBC.CDL.Integers.GreaterEqualThreshold intGreEquThr(
            threshold=numPum)
          annotation (Placement(transformation(extent={{-80,10},{-60,30}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi1
          annotation (Placement(transformation(extent={{-20,10},{0,30}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant lowMaxSpe(k=lowMax)
          "Low maximum speed"
          annotation (Placement(transformation(extent={{-80,-30},{-60,-10}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant uni(k=1)
          "full maximum speed"
          annotation (Placement(transformation(extent={{-80,50},{-60,70}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi3
          annotation (Placement(transformation(extent={{40,-50},{60,-30}})));
        Modelica.Blocks.Sources.BooleanExpression FreOrFul(y=uOpeMod == Integer(FiveZone.Types.CoolingModes.FreeCooling)
               or uOpeMod == Integer(FiveZone.Types.CoolingModes.FullMechanical))
          "Free cooling or full mechanical cooling"
          annotation (Placement(transformation(extent={{-20,-50},{0,-30}})));
        Buildings.Controls.OBC.CDL.Interfaces.RealOutput y
          "Connector of Real output signal"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi2
          annotation (Placement(transformation(extent={{-20,-80},{0,-60}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant pmcMaxSpe(k=pmcMax)
          "Maximum speed for pmc and ppmc mode"
          annotation (Placement(transformation(extent={{-80,-60},{-60,-40}})));
        Modelica.Blocks.Sources.BooleanExpression Pmc(y=uOpeMod == Integer(FiveZone.Types.CoolingModes.PartialMechanical))
          "Partial mechanical cooling"
          annotation (Placement(transformation(extent={{-80,-80},{-60,-60}})));
      equation
        connect(intGreEquThr.y, swi1.u2)
          annotation (Line(points={{-58,20},{-22,20}},   color={255,0,255}));
        connect(numActPum, intGreEquThr.u)
          annotation (Line(points={{-120,40},{-90,40},{-90,20},{-82,20}},
                                                          color={255,127,0}));
        connect(uni.y, swi1.u1) annotation (Line(points={{-58,60},{-40,60},{-40,28},{-22,
                28}},  color={0,0,127}));
        connect(lowMaxSpe.y, swi1.u3) annotation (Line(points={{-58,-20},{-40,-20},{-40,
                12},{-22,12}},   color={0,0,127}));
        connect(swi1.y, swi3.u1) annotation (Line(points={{2,20},{20,20},{20,-32},{38,
                -32}}, color={0,0,127}));
        connect(FreOrFul.y, swi3.u2)
          annotation (Line(points={{1,-40},{38,-40}}, color={255,0,255}));
        connect(swi3.y, y) annotation (Line(points={{62,-40},{80,-40},{80,0},{110,0}},
              color={0,0,127}));
        connect(pmcMaxSpe.y, swi2.u1) annotation (Line(points={{-58,-50},{-40,-50},{-40,
                -62},{-22,-62}}, color={0,0,127}));
        connect(swi2.y, swi3.u3) annotation (Line(points={{2,-70},{20,-70},{20,-48},{38,
                -48}}, color={0,0,127}));
        connect(uni.y, swi2.u3) annotation (Line(points={{-58,60},{-40,60},{-40,-78},{
                -22,-78}}, color={0,0,127}));
        connect(Pmc.y, swi2.u2)
          annotation (Line(points={{-59,-70},{-22,-70}}, color={255,0,255}));
        annotation (defaultComponentName = "maxSpeFan",
          Documentation(info="<html>
<p>
The maximum fan speed in cooling towers is reset based on cooling modes and operation status.
</p>
<ul>
<li>
When in unoccupied mode, the maximum speed is not reset.
</li>
<li>
When in free cooling mode, if all condenser pumps are enabled, the maximum fan speed is reset to full speed 100%; Otherwise the maximum fan speed is reset to a lower speed, e.g. 90%.
</li>
<li>
When in pre-partial and partial mechanical cooling mode, the maximum fan speed is set to a high speed e.g. 95%.
</li>
<li>
When in full mechanical cooling mode, if all the condenser water pumps are active, the maximum fan speed is reset to full speed 100%; Otherwise it is reset to a lower speed, e.g. 90%.
</li>
</ul>
</html>"));
      end MaximumSpeedFan;

      model StageCell "Cooling tower cell stage number control"
        extends Modelica.Blocks.Icons.Block;
        parameter Integer numCooTow = 2 "Design number of cooling towers";
        parameter Modelica.SIunits.MassFlowRate m_flow_nominal = 50
          "Nominal mass flow rate of one cooling tower";
        parameter Real staUpThr = 1.25 "Staging up threshold";
        parameter Real staDowThr = 0.55 "Staging down threshold";
        parameter Modelica.SIunits.Time waiTimStaUp=900
          "Time duration of for staging up";
        parameter Modelica.SIunits.Time waiTimStaDow=300
          "Time duration of for staging down";
        parameter Modelica.SIunits.Time shoCycTim=1200
          "Time duration to avoid short cycling of equipment";

        Modelica.Blocks.Interfaces.RealInput aveMasFlo
          "Average mass flowrate of condenser water in active cooling towers"
          annotation (Placement(
              transformation(extent={{-140,-62},{-100,-22}}),iconTransformation(
                extent={{-140,-62},{-100,-22}})));
        Modelica.Blocks.Interfaces.IntegerOutput yNumCel
          "Stage number of next time step" annotation (Placement(transformation(
                extent={{100,30},{120,50}}), iconTransformation(extent={{100,30},{120,
                  50}})));

        Modelica.Blocks.Interfaces.IntegerInput minNumCel
          "Minimum number of active tower cells determined by minimum cell controller"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}}),
              iconTransformation(extent={{-140,-20},{-100,20}})));
        BaseClasses.SequenceSignal seqSig(n=numCooTow)
          "Simple model that is used to determine the on and off sequence of equipment"
          annotation (Placement(transformation(extent={{60,-10},{80,10}})));
        Modelica.Blocks.Interfaces.RealOutput y[numCooTow]
          "On and off signal of cooling tower cell"
          annotation (Placement(transformation(extent={{100,-50},{120,-30}})));
        BaseClasses.Stage sta(
          staUpThr=staUpThr*m_flow_nominal,
          staDowThr=staDowThr*m_flow_nominal,
          waiTimStaUp=waiTimStaUp,
          waiTimStaDow=waiTimStaDow,
          shoCycTim=shoCycTim) "Stage controller"
          annotation (Placement(transformation(extent={{-40,-56},{-20,-36}})));
        Buildings.Controls.OBC.CDL.Integers.Max maxInt "Max"
          annotation (Placement(transformation(extent={{-10,-50},{10,-30}})));

        Modelica.Blocks.Interfaces.IntegerInput uOpeMod
          "Cooling mode in WSEControls.Type.OperationaModes" annotation (Placement(
              transformation(extent={{-140,20},{-100,60}}),  iconTransformation(
                extent={{-140,20},{-100,60}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi1
          annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
        Buildings.Controls.OBC.CDL.Conversions.IntegerToReal intToRea1
          annotation (Placement(transformation(extent={{22,-50},{42,-30}})));
        Modelica.Blocks.Sources.BooleanExpression unOcc(y=uOpeMod == Integer(FiveZone.Types.CoolingModes.Off))
          "Unoccupied mode"
          annotation (Placement(transformation(extent={{-80,-10},{-60,10}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant zer(k=0) "Zero"
          annotation (Placement(transformation(extent={{-50,20},{-30,40}})));
        Buildings.Controls.OBC.CDL.Conversions.RealToInteger reaToInt
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));
        Buildings.Controls.OBC.CDL.Logical.Not not2
          annotation (Placement(transformation(extent={{-74,-72},{-54,-52}})));
      equation
        connect(seqSig.y, y)
          annotation (Line(points={{81,0},{90,0},{90,-40},{110,-40}},
                                                    color={0,0,127}));
        connect(aveMasFlo, sta.u) annotation (Line(points={{-120,-42},{-42,-42}},
                          color={0,0,127}));
        connect(unOcc.y, swi1.u2)
          annotation (Line(points={{-59,0},{-12,0}}, color={255,0,255}));
        connect(minNumCel, maxInt.u1) annotation (Line(points={{-120,0},{-92,0},{-92,
                -20},{-20,-20},{-20,-34},{-12,-34}},
                                     color={255,127,0}));
        connect(intToRea1.y, swi1.u3) annotation (Line(points={{44,-40},{50,-40},{50,-16},
                {-40,-16},{-40,-8},{-12,-8}},      color={0,0,127}));
        connect(zer.y, swi1.u1) annotation (Line(points={{-28,30},{-20,30},{-20,8},{-12,
                8}},     color={0,0,127}));
        connect(seqSig.u, reaToInt.y)
          annotation (Line(points={{58,0},{42,0}}, color={255,127,0}));
        connect(swi1.y, reaToInt.u)
          annotation (Line(points={{12,0},{18,0}}, color={0,0,127}));
        connect(reaToInt.y, yNumCel) annotation (Line(points={{42,0},{50,0},{50,40},{110,
                40}},     color={255,127,0}));
        connect(sta.ySta, maxInt.u2) annotation (Line(points={{-19,-46},{-12,-46}},
                                     color={255,127,0}));
        connect(maxInt.y, intToRea1.u) annotation (Line(points={{12,-40},{20,-40}},
                               color={255,127,0}));
        connect(unOcc.y, not2.u) annotation (Line(points={{-59,0},{-50,0},{-50,-18},{
                -80,-18},{-80,-62},{-76,-62}}, color={255,0,255}));
        connect(not2.y, sta.on) annotation (Line(points={{-52,-62},{-48,-62},{-48,-50},
                {-42,-50}}, color={255,0,255}));
        annotation (defaultComponentName = "staCel",
          Documentation(info="<html>
<p>The cooling tower cell staging control is based on the water flowrate going through the cooling tower under the operation mode except the unoccuiped mode.  In the unoccupied mode, all the cells are staged off.</p>
<ul>
<li>One additional cell stages on if average flowrate through active cells is greater than a stage-up threshold <code>staUpThr*m_flow_nominal</code> for 15 minutes. </li>
<li>One additional cell stages off if average flowrate through active cells is lower than a stage-down threshold <code>staDowThr*m_flow_nominal</code> for 5 minutes. </li>
</ul>
</html>",       revisions=""),
          Diagram(coordinateSystem(extent={{-100,-80},{100,80}})),
          __Dymola_Commands);
      end StageCell;

      model SpeedFan "Cooling tower fan speed control"
        extends Modelica.Blocks.Icons.Block;

        parameter Modelica.Blocks.Types.SimpleController controllerType=
          Modelica.Blocks.Types.SimpleController.PID
          "Type of controller"
          annotation(Dialog(tab="Controller"));
        parameter Real k = 1
          "Gain of controller"
          annotation(Dialog(tab="Controller"));
        parameter Modelica.SIunits.Time Ti=0.5
          "Time constant of integrator block"
           annotation (Dialog(enable=
                (controllerType == Modelica.Blocks.Types.SimpleController.PI or
                controllerType == Modelica.Blocks.Types.SimpleController.PID),tab="Controller"));
        parameter Modelica.SIunits.Time Td(min=0)=0.1
          "Time constant of derivative block"
           annotation (Dialog(enable=
                (controllerType == Modelica.Blocks.Types.SimpleController.PD or
                controllerType == Modelica.Blocks.Types.SimpleController.PID),tab="Controller"));
        parameter Real yMax(start=1)=1
         "Upper limit of output"
          annotation(Dialog(tab="Controller"));
        parameter Real yMin=0.2
         "Lower limit of output"
          annotation(Dialog(tab="Controller"));
        parameter Boolean reverseAction = true
          "Set to true for throttling the water flow rate through a cooling coil controller"
          annotation(Dialog(tab="Controller"));
        parameter Boolean pre_y_start=false "Value of pre(y) at initial time"
          annotation(Dialog(tab="Controller"));

        Modelica.Blocks.Interfaces.RealInput TCHWSupSet(
          final quantity="ThermodynamicTemperature",
          final unit="K",
          displayUnit="degC") "Chilled water supply temperature setpoint"
          annotation (Placement(transformation(extent={{-140,40},{-100,80}}),
              iconTransformation(extent={{-140,40},{-100,80}})));
        Modelica.Blocks.Interfaces.RealInput TCWSupSet(
          final quantity="ThermodynamicTemperature",
          final unit="K",
          displayUnit="degC") "Condenser water supply temperature setpoint"
          annotation (Placement(transformation(extent={{-140,0},{-100,40}}),
              iconTransformation(extent={{-140,0},{-100,40}})));
        Modelica.Blocks.Interfaces.RealInput TCHWSup(
          final quantity="ThermodynamicTemperature",
          final unit="K",
          displayUnit="degC") "Chilled water supply temperature " annotation (
            Placement(transformation(
              extent={{-20,-20},{20,20}},
              origin={-120,-20}), iconTransformation(extent={{-140,-40},{-100,0}})));
        Buildings.Controls.Continuous.LimPID conPID(
          controllerType=controllerType,
          k=k,
          Ti=Ti,
          Td=Td,
          yMax=yMax,
          yMin=yMin,
          reverseAction=reverseAction,
          initType=Modelica.Blocks.Types.InitPID.DoNotUse_InitialIntegratorState,
          y_reset=1) "PID controller to maintain the CW/CHW supply temperature"
          annotation (Placement(transformation(extent={{0,-40},{20,-20}})));
        Modelica.Blocks.Interfaces.RealInput TCWSup(
          final quantity="ThermodynamicTemperature",
          final unit="K",
          displayUnit="degC") "Condenser water supply temperature " annotation (
            Placement(transformation(
              extent={{20,20},{-20,-20}},
              rotation=180,
              origin={-120,-60}), iconTransformation(
              extent={{20,20},{-20,-20}},
              rotation=180,
              origin={-120,-60})));

        Modelica.Blocks.Sources.Constant off(k=0) "Turn off"
          annotation (Placement(transformation(extent={{0,20},{20,40}})));
        Modelica.Blocks.Sources.BooleanExpression unOcc(y=uOpeMod == Integer(FiveZone.Types.CoolingModes.Off))
          "Unoccupied mode"
          annotation (Placement(transformation(extent={{0,-10},{20,10}})));
        Modelica.Blocks.Sources.BooleanExpression freCoo(y=uOpeMod == Integer(FiveZone.Types.CoolingModes.FreeCooling))
          "Free cooling"
          annotation (Placement(transformation(extent={{-90,50},{-70,70}})));
        Modelica.Blocks.Interfaces.IntegerInput uOpeMod "Cooling mode" annotation (
            Placement(transformation(extent={{-140,80},{-100,120}}),
              iconTransformation(extent={{-140,80},{-100,120}})));
        Modelica.Blocks.Interfaces.RealInput uFanMax "Maximum fan speed"
          annotation (Placement(transformation(extent={{-140,-120},{-100,-80}}),
              iconTransformation(extent={{-140,-120},{-100,-80}})));
        Modelica.Blocks.Interfaces.RealOutput y "Cooling tower fan speed"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));
        Modelica.Blocks.Math.Min min "Minum value"
          annotation (Placement(transformation(extent={{72,-10},{92,10}})));

      protected
        Modelica.Blocks.Logical.Switch swi1
          "The switch based on whether it is in FMC mode"
          annotation (Placement(transformation(extent={{-38,50},{-18,70}})));
        Modelica.Blocks.Logical.Switch swi2
          "The switch based on whether it is in the FMC mode"
          annotation (Placement(transformation(extent={{-10,-10},{10,10}},
              origin={-30,-52})));
        Modelica.Blocks.Logical.Switch swi3
          "The switch based on whether it is in PMC mode"
          annotation (Placement(transformation(extent={{-10,-10},{10,10}},
              origin={50,0})));

        Modelica.Blocks.Logical.Switch swi4
          "The switch based on whether it is in PMC mode"
          annotation (Placement(transformation(extent={{-10,-10},{10,10}},
              origin={50,70})));
      public
        Buildings.Controls.OBC.CDL.Logical.OnOffController onOffCon(
                         final pre_y_start=pre_y_start, final bandwidth=0.5)
                                         "Electric heater on-off controller"
          annotation (Placement(transformation(extent={{-20,94},{0,114}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant froTem(k=273.15)
          "Frozen temperature"
          annotation (Placement(transformation(extent={{-80,100},{-60,120}})));
      equation
        connect(swi1.y, conPID.u_s)
          annotation (Line(points={{-17,60},{-10,60},{-10,-30},{-2,-30}},
                       color={0,0,127}));
        connect(swi2.y, conPID.u_m)
          annotation (Line(points={{-19,-52},{10,-52},{10,-42}}, color={0,0,127}));
        connect(unOcc.y, swi3.u2)
          annotation (Line(points={{21,0},{38,0}}, color={255,0,255}));
        connect(off.y, swi3.u1)
          annotation (Line(points={{21,30},{30,30},{30,8},{38,8}}, color={0,0,127}));
        connect(conPID.y, swi3.u3)
          annotation (Line(points={{21,-30},{30,-30},{30,-8},{38,-8}},
                          color={0,0,127}));
        connect(swi3.y, min.u1)
          annotation (Line(points={{61,0},{66,0},{66,6},{70,6}}, color={0,0,127}));
        connect(min.u2,uFanMax)  annotation (Line(points={{70,-6},{64,-6},{64,-100},{-120,
                -100}}, color={0,0,127}));
        connect(TCHWSupSet, swi1.u1) annotation (Line(points={{-120,60},{-94,60},{-94,
                80},{-50,80},{-50,68},{-40,68}},
                               color={0,0,127}));
        connect(TCWSupSet, swi1.u3) annotation (Line(points={{-120,20},{-50,20},{-50,52},
                {-40,52}}, color={0,0,127}));
        connect(TCHWSup, swi2.u1) annotation (Line(points={{-120,-20},{-50,-20},{-50,-44},
                {-42,-44}}, color={0,0,127}));
        connect(TCWSup, swi2.u3)
          annotation (Line(points={{-120,-60},{-42,-60}}, color={0,0,127}));
        connect(freCoo.y, swi1.u2)
          annotation (Line(points={{-69,60},{-40,60}}, color={255,0,255}));
        connect(freCoo.y, swi2.u2) annotation (Line(points={{-69,60},{-60,60},{-60,-52},
                {-42,-52}}, color={255,0,255}));
        connect(off.y, swi4.u1) annotation (Line(points={{21,30},{30,30},{30,78},{38,
                78}}, color={0,0,127}));
        connect(min.y, swi4.u3) annotation (Line(points={{93,0},{96,0},{96,46},{32,46},
                {32,62},{38,62}}, color={0,0,127}));
        connect(swi4.y, y) annotation (Line(points={{61,70},{98,70},{98,0},{110,0}},
              color={0,0,127}));
        connect(froTem.y, onOffCon.reference)
          annotation (Line(points={{-58,110},{-22,110}}, color={0,0,127}));
        connect(TCWSup, onOffCon.u) annotation (Line(points={{-120,-60},{-94,-60},{-94,
                82},{-50,82},{-50,98},{-22,98}}, color={0,0,127}));
        connect(onOffCon.y, swi4.u2) annotation (Line(points={{2,104},{26,104},{26,70},
                {38,70}}, color={255,0,255}));
        annotation (defaultComponentName = "speFan",
        Documentation(info="<html>
<p>
Cooling tower fan speed is controlled in different ways when operation mode changes.
</p>
<ul>
<li>
For unoccupied operation mode, the fan is turned off.
</li>
<li>
For free cooling mode, the fan speed is controlled to maintain a predefined chilled water supply temperature at the downstream of the economizer, 
and not exceed the predefined maximum fan
speed. 
</li>
<li>
For pre-partial, partial and full mechanical cooling, the fan speed is controlled to maintain the supply condenser water at its setpoint. 
</li>
</ul>
</html>",       revisions=""),
          Diagram(coordinateSystem(extent={{-100,-120},{100,120}})));
      end SpeedFan;

      model SpeedPump "Pump speed control in condenser water loop"
        extends Modelica.Blocks.Icons.Block;
        parameter Modelica.SIunits.Pressure dpSetDes "Differential pressure setpoint at design condition ";

        parameter Modelica.Blocks.Types.SimpleController controllerType=Modelica.Blocks.Types.SimpleController.PID
          "Type of controller";
        parameter Real k=1 "Gain of controller";
        parameter Modelica.SIunits.Time Ti=0.5 "Time constant of Integrator block";
        parameter Modelica.SIunits.Time Td=0.1 "Time constant of Derivative block";
        parameter Real yMax=1 "Upper limit of output";
        parameter Real yMin=0.4 "Lower limit of output";
        parameter Boolean reverseAction=false
          "Set to true for throttling the water flow rate through a cooling coil controller";
        Modelica.Blocks.Interfaces.RealInput uLoa
          "Percentage of load in chillers (total loads divided by nominal capacity of all operating chillers)"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}}),
              iconTransformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Interfaces.IntegerInput uOpeMod
          "Cooling mode in WSEControls.Type.OperationaModes" annotation (Placement(
              transformation(extent={{-140,60},{-100,100}}), iconTransformation(
                extent={{-140,60},{-100,100}})));
        Modelica.Blocks.Interfaces.RealInput uSpeTow "Speed of cooling tower fans"
          annotation (Placement(transformation(extent={{-140,20},{-100,60}}),
              iconTransformation(extent={{-140,20},{-100,60}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant minPumSpe(k=yMin)
          "Minimum pump speed"
          annotation (Placement(transformation(extent={{-80,50},{-60,70}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant zer(k=0) "Zero"
          annotation (Placement(transformation(extent={{32,40},{52,60}})));
        Modelica.Blocks.Sources.BooleanExpression notOcc(y=uOpeMod == Integer(FiveZone.Types.CoolingModes.Off))
          "Not occupied"
          annotation (Placement(transformation(extent={{32,10},{52,30}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi
          annotation (Placement(transformation(extent={{70,-10},{90,10}})));
        Buildings.Controls.OBC.CDL.Continuous.Max max
          annotation (Placement(transformation(extent={{-32,20},{-12,40}})));
        Buildings.Controls.OBC.CDL.Logical.Switch swi1
          annotation (Placement(transformation(extent={{32,-18},{52,2}})));
        Modelica.Blocks.Sources.BooleanExpression freCoo(y=uOpeMod == Integer(FiveZone.Types.CoolingModes.FreeCooling))
          "Free cooling"
          annotation (Placement(transformation(extent={{0,-18},{20,2}})));
        Buildings.Controls.Continuous.LimPID con(
          controllerType=controllerType,
          k=k,
          Ti=Ti,
          Td=Td,
          yMax=yMax,
          yMin=yMin,
          reverseAction=reverseAction)           "PID controller"
          annotation (Placement(transformation(extent={{-40,-60},{-20,-40}})));
        Modelica.Blocks.Interfaces.RealInput dpSet "Differential pressure setpoint"
          annotation (Placement(transformation(extent={{-140,-60},{-100,-20}}),
              iconTransformation(extent={{-140,-60},{-100,-20}})));
        Modelica.Blocks.Interfaces.RealInput dpMea
          "Differential pressure measurement"
          annotation (Placement(transformation(extent={{-140,-100},{-100,-60}}),
              iconTransformation(extent={{-140,-100},{-100,-60}})));
        Buildings.Controls.OBC.CDL.Continuous.Gain gai1(k=1/dpSetDes)
          annotation (Placement(transformation(extent={{-80,-40},{-60,-20}})));
        Buildings.Controls.OBC.CDL.Continuous.Gain gai2(k=1/dpSetDes)
          annotation (Placement(transformation(extent={{-80,-90},{-60,-70}})));
        Buildings.Utilities.Math.Max max2(nin=3)
          annotation (Placement(transformation(extent={{0,-50},{20,-30}})));

        Buildings.Controls.OBC.CDL.Interfaces.RealOutput y "Speed signal"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));
      equation
        connect(notOcc.y, swi.u2) annotation (Line(points={{53,20},{58,20},{58,0},{68,
                0}},  color={255,0,255}));
        connect(zer.y, swi.u1) annotation (Line(points={{54,50},{60,50},{60,8},{68,8}},
              color={0,0,127}));
        connect(minPumSpe.y,max. u1) annotation (Line(points={{-58,60},{-50,60},{-50,36},
                {-34,36}}, color={0,0,127}));
        connect(uSpeTow,max. u2) annotation (Line(points={{-120,40},{-86,40},{-86,24},
                {-34,24}}, color={0,0,127}));
        connect(max.y, swi1.u1) annotation (Line(points={{-10,30},{20,30},{20,0},{30,0}},
              color={0,0,127}));
        connect(freCoo.y, swi1.u2) annotation (Line(points={{21,-8},{30,-8}},
                      color={255,0,255}));
        connect(gai1.y, con.u_s) annotation (Line(points={{-58,-30},{-50,-30},{-50,-50},
                {-42,-50}}, color={0,0,127}));
        connect(dpSet, gai1.u)
          annotation (Line(points={{-120,-40},{-90,-40},{-90,-30},{-82,-30}},
                                                          color={0,0,127}));
        connect(dpMea, gai2.u) annotation (Line(points={{-120,-80},{-102,-80},{-102,
                -80},{-82,-80}},
                            color={0,0,127}));
        connect(gai2.y, con.u_m)
          annotation (Line(points={{-58,-80},{-30,-80},{-30,-62}}, color={0,0,127}));
        connect(minPumSpe.y,max2. u[1]) annotation (Line(points={{-58,60},{-50,
                60},{-50,-18},{-14,-18},{-14,-41.3333},{-2,-41.3333}},
                                                          color={0,0,127}));
        connect(uLoa,max2. u[2]) annotation (Line(points={{-120,0},{-52,0},{-52,-20},
                {-16,-20},{-16,-40},{-2,-40}},color={0,0,127}));
        connect(con.y,max2. u[3]) annotation (Line(points={{-19,-50},{-12,-50},
                {-12,-38.6667},{-2,-38.6667}},
                                color={0,0,127}));
        connect(max2.y, swi1.u3) annotation (Line(points={{21,-40},{24,-40},{24,-16},
                {30,-16}},
                      color={0,0,127}));
        connect(swi1.y, swi.u3) annotation (Line(points={{54,-8},{68,-8}},
              color={0,0,127}));
        connect(swi.y, y) annotation (Line(points={{92,0},{110,0}}, color={0,0,127}));
        annotation (defaultComponentName="spePum", Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          Documentation(info="<html>
<p>
Condenser water pump speed control is different in different operation modes.
</p>
<ul>
<li>
For unoccupied operation mode, the pump is turned off.
</li>
<li>
For free cooling mode, the condenser water pump speed is equal to a high signal select of a PID loop output and a minimum speed (e.g. 40%). The PID loop outputs the cooling tower
fan speed signal to maintain chilled water supply temperature at its setpoint. 
</li>
<li>
For pre-partial, partial and full mechanical cooling, the condenser water pump speed is equal to a high signal select of the following three: (1) a minimum speed (e.g. 40%); (2) highest chiller percentage load; 
(3) CW system differential pressure PID output signal. 
</li>
</ul>
</html>"));
      end SpeedPump;

      model SupplyTemperatureReset
        "Cooling tower supply temperature setpoint reset"
        extends Modelica.Blocks.Icons.Block;
        parameter Modelica.SIunits.ThermodynamicTemperature TSetMinFulMec = 273.15 + 12.78
        "Minimum cooling tower supply temperature setpoint for full mechanical cooling";
        parameter Modelica.SIunits.ThermodynamicTemperature TSetMaxFulMec = 273.15 + 35
        "Maximum cooling tower supply temperature setpoint for full mechanical cooling";
        parameter Modelica.SIunits.ThermodynamicTemperature TSetParMec = 273.15 + 10
        "Cooling tower supply temperature setpoint for partial mechanical cooling";
        Modelica.Blocks.Interfaces.IntegerInput uOpeMod
          "Cooling mode signal, integer value of WSEControlLogics.Controls.WSEControls.Type.OperationModes"
          annotation (
            Placement(transformation(extent={{-140,30},{-100,70}}),
              iconTransformation(extent={{-140,30},{-100,70}})));
        Modelica.Blocks.Interfaces.RealOutput TSet(
          final quantity="ThermodynamicTemperature",
          final unit="K",
          displayUnit="degC") "Temperature setpoint" annotation (
           Placement(transformation(extent={{100,-10},{120,10}}), iconTransformation(
                extent={{100,-10},{120,10}})));

        Modelica.Blocks.Sources.BooleanExpression fmcMod(y=uOpeMod == Integer(FiveZone.Types.CoolingModes.FullMechanical))
          "Full mechanical cooling mode"
          annotation (Placement(transformation(extent={{0,-30},{20,-10}})));

        Modelica.Blocks.Interfaces.RealInput TWetBul(
          final quantity="ThermodynamicTemperature",
          final unit="K",
          displayUnit="degC") "Outdoor air wet bulb emperature" annotation (Placement(
              transformation(extent={{-140,-20},{-100,20}}),iconTransformation(extent={{-140,
                  -20},{-100,20}})));
        Modelica.Blocks.Interfaces.RealInput TAppCooTow(
          final quantity="ThermodynamicTemperature",
          final unit="K",
          displayUnit="degC") "Approach temperature in cooling towers" annotation (
            Placement(transformation(extent={{-140,-70},{-100,-30}}),
              iconTransformation(extent={{-140,-70},{-100,-30}})));
        Buildings.Controls.OBC.CDL.Continuous.Add add1 "Addition"
          annotation (Placement(transformation(extent={{-80,-10},{-60,10}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant con1(k(unit="K")=
            TSetParMec)
          annotation (Placement(transformation(extent={{0,-68},{20,-48}})));

      protected
        Modelica.Blocks.Logical.Switch swi1
          "The switch based on whether it is in FMC"
          annotation (Placement(transformation(extent={{60,-10},{80,10}})));

      public
        Modelica.Blocks.Math.Min min
          annotation (Placement(transformation(extent={{-40,30},{-20,50}})));
        Modelica.Blocks.Math.Max max
          annotation (Placement(transformation(extent={{0,0},{20,20}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant con2(k(unit="K")=
            TSetMinFulMec)
          annotation (Placement(transformation(extent={{-40,-30},{-20,-10}})));
        Buildings.Controls.OBC.CDL.Continuous.Sources.Constant con3(k(unit="K")=
            TSetMaxFulMec)
          annotation (Placement(transformation(extent={{-80,60},{-60,80}})));
      equation
        connect(fmcMod.y, swi1.u2)
          annotation (Line(points={{21,-20},{38,-20},{38,0},{58,0}},
                                                    color={255,0,255}));
        connect(swi1.y, TSet)
          annotation (Line(points={{81,0},{110,0}}, color={0,0,127}));
        connect(TWetBul, add1.u1) annotation (Line(points={{-120,0},{-94,0},{-94,6},{-82,
                6}},       color={0,0,127}));
        connect(TAppCooTow, add1.u2) annotation (Line(points={{-120,-50},{-90,-50},{-90,
                -6},{-82,-6}},
                           color={0,0,127}));
        connect(con1.y, swi1.u3) annotation (Line(points={{22,-58},{40,-58},{40,-8},{58,
                -8}}, color={0,0,127}));
        connect(con3.y, min.u1) annotation (Line(points={{-58,70},{-52,70},{-52,46},{-42,
                46}}, color={0,0,127}));
        connect(add1.y, min.u2) annotation (Line(points={{-58,0},{-52,0},{-52,34},{-42,
                34}}, color={0,0,127}));
        connect(min.y, max.u1) annotation (Line(points={{-19,40},{-12,40},{-12,16},{-2,
                16}}, color={0,0,127}));
        connect(con2.y, max.u2) annotation (Line(points={{-18,-20},{-12,-20},{-12,4},{
                -2,4}}, color={0,0,127}));
        connect(max.y, swi1.u1)
          annotation (Line(points={{21,10},{40,10},{40,8},{58,8}}, color={0,0,127}));
       annotation (defaultComponentName="temRes", Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          Documentation(info="<html>
<p>This model describes a cooling tower supply temperature reset for a chilled water system with integrated waterside economizers.</p>
<ul>
<li>When in unoccupied mode, the condenser supply temperature is free floated, and keep unchanged from previous mode</li>
<li>When in free cooling, the condenser water supply temperature is free floated, and keep unchanged from previous mode</li>
<li>When in pre-partial, and partial mechanical cooling, the condenser water supply temperature is reset to a predefined value <code>TSetParMec</code>.This could be changed
based on advanced control algorithm.</li>
<li>When in full mechanical cooling mode, the condenser water supply temperature is reset according to the environment.
 <i>T<sub>sup,CW,set</sub> = T<sub>wb,OA</sub> + T<sub>app,pre</sub></i>. T<sub>sup,CW,set</sub> means the supply condenser water temperature setpoint, T<sub>wb,OA</sub>
is the outdoor air wet bulb temperature, and T<sub>app,pre</sub> is the predicted approach temperature, which could be a fixed or various value.</li>
</ul>
</html>"));
      end SupplyTemperatureReset;

      annotation (Documentation(info="<html>
<p>This package contains a collection of the local controls in the condenser water loop.</p>
</html>"));
    end CWLoopEquipment;

    package BaseClasses "Base classes for local controls of the chilled water system with water economizer"

      model LinearMap "Ratio function"
        extends Modelica.Blocks.Interfaces.SISO;
        parameter Boolean use_uInpRef1_in = false "True if use outside values for uInpRef1";
        parameter Boolean use_uInpRef2_in = false "True if use outside values for uInpRef2";
        parameter Boolean use_yOutRef1_in = false "True if use outside values for uOutRef1";
        parameter Boolean use_yOutRef2_in = false "True if use outside values for uOutRef2";
        parameter Real uInpRef1= 0 "Minimum limit"
          annotation(Dialog(enable = not use_uInpRef1_in));
        parameter Real uInpRef2= 1 "Maximum limit"
          annotation(Dialog(enable = not use_uInpRef2_in));
        parameter Real yOutRef1= 0 "Minimum limit"
          annotation(Dialog(enable = not use_yOutRef1_in));
        parameter Real yOutRef2= 1 "Maximum limit"
          annotation(Dialog(enable = not use_yOutRef2_in));
        parameter Real dy= 1e-3 "Transition interval";

        Modelica.Blocks.Interfaces.RealInput uInpRef1_in if use_uInpRef1_in "Connector of Real input signal"
          annotation (Placement(transformation(extent={{-140,60},{-100,100}})));
        Modelica.Blocks.Interfaces.RealInput uInpRef2_in if use_uInpRef2_in "Connector of Real input signal"
          annotation (Placement(transformation(extent={{-140,20},{-100,60}})));

        Modelica.Blocks.Interfaces.RealInput yOutRef2_in if use_yOutRef2_in "Connector of Real input signal"
          annotation (Placement(transformation(extent={{-140,-100},{-100,-60}})));

        Modelica.Blocks.Interfaces.RealInput yOutRef1_in if use_yOutRef1_in "Connector of Real input signal"
          annotation (Placement(transformation(extent={{-140,-60},{-100,-20}})));

      protected
        Real outInt "Intermediate output";
        Modelica.Blocks.Interfaces.RealInput y1;
        Modelica.Blocks.Interfaces.RealInput y2;
        Modelica.Blocks.Interfaces.RealInput u1;
        Modelica.Blocks.Interfaces.RealInput u2;

      equation
        connect(u1,uInpRef1_in);
        connect(u2,uInpRef2_in);
        connect(y1,yOutRef1_in);
        connect(y2,yOutRef2_in);

        if not use_uInpRef1_in then
          u1 = uInpRef1;
        end if;
        if not use_uInpRef2_in then
          u2 = uInpRef2;
        end if;
        if not use_yOutRef1_in then
          y1 = yOutRef1;
        end if;
        if not use_yOutRef2_in then
          y2 = yOutRef2;
        end if;

        outInt = y1 + (u - u1)*(y2 - y1)/(u2 - u1);

        y=Buildings.Utilities.Math.Functions.smoothLimit(
                     outInt,min(y1,y2),max(y1,y2),dy);

        annotation (defaultComponentName = "linMap",
        Icon(graphics={Text(
              extent={{-98,24},{100,-12}},
              lineColor={238,46,47},
                textString="%name")}));
      end LinearMap;

      block LinearPiecewiseTwo
        "A two-pieces linear piecewise function"
        extends Modelica.Blocks.Icons.Block;
        parameter Real x0 "First interval [x0, x1]";
        parameter Real x1 "First interval [x0, x1] and second interval (x1, x2]";
        parameter Real x2 "Second interval (x1, x2]";
        parameter Real y10 "y[1] at u = x0";
        parameter Real y11 "y[1] at u = x1";
        parameter Real y20 "y[2] at u = x1";
        parameter Real y21 "y[2] at u = x2";
        Modelica.Blocks.Interfaces.RealInput u "Set point" annotation (extent=[-190,
              80; -150, 120], Placement(transformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Interfaces.RealOutput y[2] "Connectors of Real output signal"
          annotation (extent=[148, -10; 168, 10], Placement(transformation(extent={{100,-10},
                  {120,10}})));
        Buildings.Controls.SetPoints.Table y1Tab(table=[x0, y10; x1, y11; x2, y11])
          "Table for y[1]"
          annotation (Placement(transformation(extent={{-40,20},{-20,40}})));
        Buildings.Controls.SetPoints.Table y2Tab(table=[x0, y20; x1, y20; x2, y21])
          "Table for y[2]"
          annotation (Placement(transformation(extent={{-40,-40},{-20,-20}})));
      equation
        connect(u, y1Tab.u) annotation (Line(
            points={{-120,1.11022e-15},{-58,1.11022e-15},{-58,30},{-42,30}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(u, y2Tab.u) annotation (Line(
            points={{-120,1.11022e-15},{-58,1.11022e-15},{-58,-30},{-42,-30}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(y1Tab.y, y[1]) annotation (Line(
            points={{-19,30},{26,30},{26,-5},{110,-5}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(y2Tab.y, y[2]) annotation (Line(
            points={{-19,-30},{42,-30},{42,5},{110,5}},
            color={0,0,127},
            smooth=Smooth.None));
        annotation (
          defaultComponentName="linPieTwo",
          Documentation(info="<HTML>
<p>
This component calcuates the output according to two piecewise linear function as
</p>
<table>
<tr>
<td>
<i>u &isin; [x<sub>0</sub>, x<sub>1</sub>]:</i></td>
    <td><i>y<sub>1</sub> = y<sub>10</sub> + u (y<sub>11</sub>-y<sub>10</sub>)/(x<sub>1</sub>-x<sub>0</sub>)</i><br/>
        <i>y<sub>2</sub> = y<sub>20</sub></i></td>
</tr>
<tr>
<td><i>u &isin; (x<sub>1</sub>, x<sub>2</sub>]:</i></td>
    <td><i>y<sub>1</sub> = y<sub>11</sub></i><br/>
    <i>y<sub>2</sub> = y<sub>20</sub> + (u-x<sub>1</sub>)
       (y<sub>21</sub>-y<sub>20</sub>)/(x<sub>2</sub>-x<sub>1</sub>)</i></td>
</tr>
</table>
</HTML>",       revisions="<html>
<ul>
<li>
July 20, 2011, by Wangda Zuo:<br/>
Add comments and merge to library.
</li>
<li>
January 18, 2011, by Wangda Zuo:<br/>
First implementation.
</li>
</ul>
</html>"),Icon(graphics={
              Line(
                points={{-68,62},{-68,-50},{62,-50}},
                color={0,0,0},
                smooth=Smooth.None,
                arrow={Arrow.Filled,Arrow.Filled}),
              Line(
                points={{46,-50},{46,62}},
                color={0,0,0},
                smooth=Smooth.None,
                arrow={Arrow.None,Arrow.Filled}),
              Text(
                extent={{-52,6},{-42,-2}},
                lineColor={0,0,0},
                textString="y[1]"),
              Text(
                extent={{24,6},{34,-2}},
                lineColor={128,0,255},
                textString="y[2]",
                lineThickness=1),
              Text(
                extent={{-74,-52},{-64,-60}},
                lineColor={0,0,0},
                textString="x0"),
              Text(
                extent={{-18,-52},{-8,-60}},
                lineColor={0,0,0},
                textString="x1"),
              Text(
                extent={{40,-52},{50,-60}},
                lineColor={0,0,0},
                textString="x2"),
              Text(
                extent={{-80,-38},{-70,-46}},
                lineColor={0,0,0},
                textString="y10"),
              Text(
                extent={{-80,34},{-68,26}},
                lineColor={0,0,0},
                textString="y11"),
              Text(
                extent={{48,50},{60,42}},
                lineColor={128,0,255},
                textString="y21"),
              Text(
                extent={{48,-32},{58,-40}},
                lineColor={128,0,255},
                textString="y20",
                lineThickness=1),
              Line(
                points={{-68,-42},{-14,30},{46,30}},
                color={0,0,0},
                smooth=Smooth.None,
                thickness=1),
              Line(
                points={{-68,44},{-14,44},{46,-36}},
                color={128,0,255},
                thickness=1,
                smooth=Smooth.None),
              Line(
                points={{-14,44},{-14,-50}},
                color={175,175,175},
                smooth=Smooth.None,
                pattern=LinePattern.Dash),
              Line(
                points={{-68,30},{-14,30}},
                color={175,175,175},
                pattern=LinePattern.Dash,
                smooth=Smooth.None),
              Line(
                points={{-14,44},{46,44}},
                color={175,175,175},
                pattern=LinePattern.Dash,
                smooth=Smooth.None),
              Text(
                extent={{62,-46},{72,-54}},
                lineColor={0,0,0},
                textString="x")}));
      end LinearPiecewiseTwo;

      model SequenceSignal "Signal for each cell"
        extends Modelica.Blocks.Icons.Block;
        parameter Integer n(min=1) "Length of the signal";

        Modelica.Blocks.Interfaces.RealOutput y[n]
          "On/off signals for each equipment (1: on, 0: off)"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));

        Modelica.Blocks.Interfaces.IntegerInput u
          "Number of active tower cells" annotation (Placement(transformation(extent={{-140,
                  -20},{-100,20}}),      iconTransformation(extent={{-140,-20},{-100,20}})));

      algorithm
        y := fill(0,n);
        for i in 1:u loop
          y[i] := 1;
        end for;

        annotation (defaultComponentName = "seqSig",
        Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          Documentation(info="<html>
<p>
Simple model that is used to determine the on and off sequence of equipment. This model can be replaced by rotation control models.
The logic in this model is explained as follows:
</p>
<ul>
<li>
Among <code>n</code> equipment, the first <code>u</code> equipment are switched on, and the rest <code>n-u</code> are switched off.
</li>
</ul>
</html>"));
      end SequenceSignal;

      model Stage "General stage control model"
        extends Modelica.Blocks.Icons.Block;

        parameter Real staUpThr = 1.25 "Staging up threshold";
        parameter Real staDowThr = 0.55 "Staging down threshold";

        parameter Modelica.SIunits.Time waiTimStaUp=300
          "Time duration of for staging up";
        parameter Modelica.SIunits.Time waiTimStaDow=600
          "Time duration of for staging down";
        parameter Modelica.SIunits.Time shoCycTim=1200
          "Time duration to avoid short cycling of equipment";

        Modelica.Blocks.Interfaces.RealInput u "Measured signal" annotation (
            Placement(transformation(extent={{-140,20},{-100,60}}),
              iconTransformation(extent={{-140,20},{-100,60}})));
        Modelica.Blocks.Interfaces.IntegerOutput ySta(start=0)
          "Stage number of next time step" annotation (Placement(transformation(
                extent={{100,-10},{120,10}}), iconTransformation(extent={{100,-10},{120,
                  10}})));

        Modelica.StateGraph.InitialStepWithSignal off "All off" annotation (Placement(
              transformation(
              extent={{-10,10},{10,-10}},
              rotation=-90,
              origin={-22,50})));
        Modelica.StateGraph.StepWithSignal oneOn(nIn=2, nOut=2)
          "One equipment is staged" annotation (Placement(transformation(
              extent={{-10,10},{10,-10}},
              rotation=-90,
              origin={-22,-10})));
        Modelica.StateGraph.StepWithSignal twoOn "Two equipment are staged"
          annotation (Placement(transformation(
              extent={{-10,10},{10,-10}},
              rotation=-90,
              origin={-22,-90})));
        Modelica.StateGraph.Transition tra1(
          condition=(timGreEqu.y >= waiTimStaUp and offTim.y >= shoCycTim) or on)
           "Transition 1" annotation (Placement(
              transformation(
              extent={{-10,-10},{10,10}},
              rotation=-90,
              origin={-52,20})));
        Modelica.StateGraph.Transition tra2(
          condition=timGreEqu.y >= waiTimStaUp and oneOnTim.y >= shoCycTim)
          "Transition 1"
           annotation (Placement(
              transformation(
              extent={{-10,-10},{10,10}},
              rotation=-90,
              origin={-42,-50})));
        FiveZone.PrimarySideControl.BaseClasses.TimerGreatEqual timGreEqu(threshold=
             staUpThr) "Timer"
          annotation (Placement(transformation(extent={{-80,70},{-60,90}})));
        FiveZone.PrimarySideControl.BaseClasses.TimeLessEqual timLesEqu(threshold=
             staDowThr) "Timer"
          annotation (Placement(transformation(extent={{-80,40},{-60,60}})));
        Modelica.StateGraph.Transition tra3(condition=(timLesEqu.y >= waiTimStaDow
               and twoOnTim.y >= shoCycTim) or not on)
         "Transition 1" annotation (Placement(
              transformation(
              extent={{10,10},{-10,-10}},
              rotation=-90,
              origin={-2,-50})));
        Modelica.StateGraph.Transition tra4(
          enableTimer=false, condition=(
              timLesEqu.y >= waiTimStaDow and oneOnTim.y >= shoCycTim) or not on)
         "Transition 1" annotation (
            Placement(transformation(
              extent={{10,10},{-10,-10}},
              rotation=-90,
              origin={0,20})));
        FiveZone.PrimarySideControl.BaseClasses.Timer offTim
          "Timer for the state where equipment is off"
          annotation (Placement(transformation(extent={{18,40},{38,60}})));
        FiveZone.PrimarySideControl.BaseClasses.Timer oneOnTim
          "Timer for the state where only one equipment is on"
          annotation (Placement(transformation(extent={{18,-20},{38,0}})));
        FiveZone.PrimarySideControl.BaseClasses.Timer twoOnTim
          "Timer for the state where two equipment are on"
          annotation (Placement(transformation(extent={{18,-100},{38,-80}})));
        Modelica.Blocks.MathInteger.MultiSwitch mulSwi(expr={0,1,2}, nu=3)
          annotation (Placement(transformation(extent={{60,-10},{80,10}})));
        Modelica.Blocks.Interfaces.BooleanInput on
          "Set to true to enable equipment, or false to disable equipment"
          annotation (Placement(transformation(extent={{-140,-60},{-100,-20}}),
              iconTransformation(extent={{-140,-60},{-100,-20}})));
      equation
        connect(u, timGreEqu.u)
          annotation (Line(points={{-120,40},{-92,40},{-92,80},{-82,80}},
                                                        color={0,0,127}));
        connect(u, timLesEqu.u) annotation (Line(points={{-120,40},{-92,40},{-92,50},
                {-82,50}},color={0,0,127}));
        connect(off.active, offTim.u) annotation (Line(points={{-11,50},{16,50}},
                              color={255,0,255}));
        connect(oneOn.active, oneOnTim.u)
          annotation (Line(points={{-11,-10},{16,-10}},
                                                     color={255,0,255}));
        connect(twoOn.active, twoOnTim.u)
          annotation (Line(points={{-11,-90},{16,-90}},color={255,0,255}));
        connect(off.outPort[1], tra1.inPort) annotation (Line(points={{-22,39.5},{-22,
                34},{-52,34},{-52,24}},
                                    color={0,0,0}));
        connect(tra1.outPort, oneOn.inPort[1]) annotation (Line(points={{-52,18.5},{-52,
                8},{-22.5,8},{-22.5,1}},  color={0,0,0}));
        connect(oneOn.outPort[1], tra2.inPort) annotation (Line(points={{-22.25,-20.5},
                {-22.25,-30},{-42,-30},{-42,-46}},
                                                 color={0,0,0}));
        connect(tra2.outPort, twoOn.inPort[1]) annotation (Line(points={{-42,-51.5},{-42,
                -68},{-22,-68},{-22,-79}},
                                       color={0,0,0}));
        connect(twoOn.outPort[1], tra3.inPort) annotation (Line(points={{-22,-100.5},{
                -22,-112},{-2,-112},{-2,-54}},
                                    color={0,0,0}));
        connect(tra3.outPort, oneOn.inPort[2]) annotation (Line(points={{-2,-48.5},{-2,
                8},{-21.5,8},{-21.5,1}},color={0,0,0}));
        connect(oneOn.outPort[2], tra4.inPort) annotation (Line(points={{-21.75,-20.5},
                {-21.75,-30},{0,-30},{0,16}},
                                        color={0,0,0}));
        connect(tra4.outPort, off.inPort[1])
          annotation (Line(points={{0,21.5},{0,66},{-22,66},{-22,61}},
                                                                     color={0,0,0}));
        connect(mulSwi.y, ySta)
          annotation (Line(points={{80.5,0},{110,0}}, color={255,127,0}));
        connect(off.active, mulSwi.u[1]) annotation (Line(points={{-11,50},{8,50},{8,
                70},{54,70},{54,2},{60,2}}, color={255,0,255}));
        connect(oneOn.active, mulSwi.u[2]) annotation (Line(points={{-11,-10},{10,-10},
                {10,-28},{54,-28},{54,0},{60,0}}, color={255,0,255}));
        connect(twoOn.active, mulSwi.u[3]) annotation (Line(points={{-11,-90},{10,-90},
                {10,-110},{56,-110},{56,-2},{60,-2}}, color={255,0,255}));
        annotation (defaultComponentName = "sta",
          Documentation(info="<html>
<p>
General stage control for two equipment using state-graph package in Modelica.
</p>
<ul>
<li>
One additional equipment stages on if measured signal is greater than a stage-up threshold <code>staUpThr</code> for a predefined time period 
<code>waiTimStaUp</code>, and the elapsed time since the staged equipment was off is greater than <code>shoCycTim</code> to avoid short cycling.
</li>
<li>
One additional equipment stages off if measured signal is less than a stage-down threshold <code>staUpThr</code> for a predefined time period 
<code>waiTimStaDow</code>, and the elapsed time since the staged equipment was on is greater than <code>shoCycTim</code> to avoid short cycling.
</li>
</ul>
</html>",       revisions=""),
          Diagram(coordinateSystem(extent={{-100,-140},{100,100}})),
          __Dymola_Commands);
      end Stage;

      model StageBackup
        "General stage control model as a back model which needs to be improved and tested"
        extends Modelica.Blocks.Icons.Block;
        parameter Integer numSta "Design number of equipment that can be staged";
        parameter Real staUpThr = 1.25 "Staging up threshold";
        parameter Real staDowThr = 0.55 "Staging down threshold";

        parameter Modelica.SIunits.Time waiTimStaUp = 900 "Time duration of TFlo1 condition for staging on one tower cell";
        parameter Modelica.SIunits.Time waiTimStaDow = 300 "Time duration of TFlo2 condition for staging off one tower cell";

        Modelica.Blocks.Interfaces.RealInput u "Measured signal" annotation (
            Placement(transformation(extent={{-140,60},{-100,100}}),
              iconTransformation(extent={{-140,60},{-100,100}})));
        Modelica.Blocks.Interfaces.IntegerOutput ySta
          "Stage number of next time step" annotation (Placement(transformation(
                extent={{100,-10},{120,10}}), iconTransformation(extent={{100,-10},{120,
                  10}})));

        Modelica.Blocks.Interfaces.IntegerInput minSta "Minimum number of stages"
          annotation (Placement(transformation(extent={{-140,-100},{-100,-60}}),
              iconTransformation(extent={{-140,-100},{-100,-60}})));
        Modelica.Blocks.Interfaces.IntegerInput uSta "Number of active stages"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}}),
              iconTransformation(extent={{-140,-20},{-100,20}})));
        Buildings.Controls.OBC.CDL.Conversions.IntegerToReal intToRea
          annotation (Placement(transformation(extent={{-80,50},{-60,70}})));
        BaseClasses.TimerGreatEqual timGreEqu(threshold=staUpThr)
                                              "Timer"
          annotation (Placement(transformation(extent={{-20,70},{0,90}})));
        BaseClasses.TimeLessEqual timLesEqu(threshold=staDowThr)
                                            "Timer"
          annotation (Placement(transformation(extent={{-40,-70},{-20,-50}})));
        Buildings.Controls.OBC.CDL.Continuous.GreaterEqualThreshold staUpAct(threshold=
             waiTimStaUp)
          "Stageup activated"
          annotation (Placement(transformation(extent={{40,70},{60,90}})));
        Buildings.Controls.OBC.CDL.Continuous.GreaterEqualThreshold staDowAct(threshold=
             waiTimStaDow)
          "Stage down activated"
          annotation (Placement(transformation(extent={{20,-70},{40,-50}})));
      protected
        Modelica.Blocks.Logical.Switch swi1
          "The switch based on whether it is in FMC mode"
          annotation (Placement(transformation(extent={{20,12},{40,32}})));
      public
        Buildings.Controls.OBC.CDL.Continuous.Add add1
          annotation (Placement(transformation(extent={{-40,20},{-20,40}})));
        Modelica.Blocks.Sources.Constant uni(k=1) "One"
          annotation (Placement(transformation(extent={{-80,14},{-60,34}})));
        Buildings.Controls.OBC.CDL.Continuous.Add add2(k2=-1)
          annotation (Placement(transformation(extent={{-40,-30},{-20,-10}})));
      protected
        Modelica.Blocks.Logical.Switch swi2
          "The switch based on whether it is in FMC mode"
          annotation (Placement(transformation(extent={{8,-38},{28,-18}})));
      public
        Buildings.Controls.OBC.CDL.Conversions.RealToInteger reaToInt
          annotation (Placement(transformation(extent={{-80,-108},{-60,-88}})));
        Buildings.Controls.OBC.CDL.Integers.Max maxInt
          annotation (Placement(transformation(extent={{-40,-114},{-20,-94}})));
        Buildings.Controls.OBC.CDL.Integers.Min minInt
          annotation (Placement(transformation(extent={{20,-130},{40,-110}})));
        Buildings.Controls.OBC.CDL.Integers.Sources.Constant conInt(k=numSta)
          annotation (Placement(transformation(extent={{-40,-140},{-20,-120}})));
      equation
        connect(uSta, intToRea.u) annotation (Line(points={{-120,0},{-90,0},{-90,60},{
                -82,60}}, color={255,127,0}));
        connect(timGreEqu.y, staUpAct.u)
          annotation (Line(points={{1,80},{38,80}},  color={0,0,127}));
        connect(timLesEqu.y, staDowAct.u)
          annotation (Line(points={{-19,-60},{18,-60}},color={0,0,127}));
        connect(intToRea.y, add1.u1) annotation (Line(points={{-58,60},{-54,60},
                {-54,36},{-42,36}},
                           color={0,0,127}));
        connect(uni.y, add1.u2)
          annotation (Line(points={{-59,24},{-42,24}}, color={0,0,127}));
        connect(add1.y, swi1.u1)
          annotation (Line(points={{-18,30},{18,30}}, color={0,0,127}));
        connect(staUpAct.y, swi1.u2) annotation (Line(points={{62,80},{80,80},{
                80,54},{0,54},{0,22},{18,22}},
                                        color={255,0,255}));
        connect(intToRea.y, swi1.u3) annotation (Line(points={{-58,60},{-52,60},
                {-52,14},{18,14}},
                          color={0,0,127}));
        connect(intToRea.y, add2.u1) annotation (Line(points={{-58,60},{-50,60},
                {-50,-14},{-42,-14}},
                            color={0,0,127}));
        connect(staDowAct.y, swi2.u2) annotation (Line(points={{42,-60},{50,-60},
                {50,-44},{-6,-44},{-6,-28},{6,-28}},
                                           color={255,0,255}));
        connect(add2.y, swi2.u1)
          annotation (Line(points={{-18,-20},{6,-20}},  color={0,0,127}));
        connect(uni.y, add2.u2) annotation (Line(points={{-59,24},{-54,24},{-54,-26},{
                -42,-26}}, color={0,0,127}));
        connect(swi1.y, swi2.u3) annotation (Line(points={{41,22},{50,22},{50,0},{-10,
                0},{-10,-36},{6,-36}},   color={0,0,127}));
        connect(swi2.y, reaToInt.u) annotation (Line(points={{29,-28},{60,-28},{60,-80},
                {-92,-80},{-92,-98},{-82,-98}},    color={0,0,127}));
        connect(minSta, maxInt.u2) annotation (Line(points={{-120,-80},{-94,-80},{-94,
                -110},{-42,-110}}, color={255,127,0}));
        connect(reaToInt.y, maxInt.u1)
          annotation (Line(points={{-58,-98},{-42,-98}},color={255,127,0}));
        connect(maxInt.y, minInt.u1) annotation (Line(points={{-18,-104},{-10,
                -104},{-10,-114},{18,-114}},
                                  color={255,127,0}));
        connect(conInt.y, minInt.u2) annotation (Line(points={{-18,-130},{-10,
                -130},{-10,-126},{18,-126}},
                                  color={255,127,0}));
        connect(minInt.y, ySta) annotation (Line(points={{42,-120},{80,-120},{
                80,0},{110,0}},
                     color={255,127,0}));
        connect(u, timGreEqu.u)
          annotation (Line(points={{-120,80},{-22,80}}, color={0,0,127}));
        connect(u, timLesEqu.u) annotation (Line(points={{-120,80},{-48,80},{-48,-60},
                {-42,-60}}, color={0,0,127}));
        annotation (defaultComponentName = "sta",
          Documentation(info="<html>
<p>
The cooling tower cell staging control is based on the water flowrate going through the cooling tower.
</p>
<ul>
<li>
One additional cell stages on if average flowrate through active cells is greater than a stage-up threshold <code>staUpThr*m_flow_nominal</code> for 15 minutes.
</li>
<li>
One additional cell stages off if average flowrate through active cells is lower than a stage-down threshold <code>staDowThr*m_flow_nominal</code> for 5 minutes.
</li>
</ul>
</html>",       revisions=""),
          Diagram(coordinateSystem(extent={{-100,-140},{100,100}})),
          __Dymola_Commands);
      end StageBackup;

      block Timer
        "Timer measuring the time from the time instant where the Boolean input became true"

        extends Modelica.Blocks.Icons.PartialBooleanBlock;
        Modelica.Blocks.Interfaces.BooleanInput u "Connector of Boolean input signal"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Interfaces.RealOutput y "Connector of Real output signal"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));

      protected
        discrete Modelica.SIunits.Time entryTime "Time instant when u became true";
      initial equation
        pre(entryTime) = 0;
      equation
        when u then
          entryTime = time;
        end when;
        y = if u then time - entryTime else 0.0;
        annotation (
          Icon(
            coordinateSystem(preserveAspectRatio=true,
              extent={{-100.0,-100.0},{100.0,100.0}}),
              graphics={
            Line(points={{-90.0,-70.0},{82.0,-70.0}},
              color={192,192,192}),
            Line(points={{-80.0,68.0},{-80.0,-80.0}},
              color={192,192,192}),
            Polygon(lineColor={192,192,192},
              fillColor={192,192,192},
              fillPattern=FillPattern.Solid,
              points={{90.0,-70.0},{68.0,-62.0},{68.0,-78.0},{90.0,-70.0}}),
            Polygon(lineColor={192,192,192},
              fillColor={192,192,192},
              fillPattern=FillPattern.Solid,
              points={{-80.0,90.0},{-88.0,68.0},{-72.0,68.0},{-80.0,90.0}}),
            Line(points={{-80.0,-70.0},{-60.0,-70.0},{-60.0,-26.0},{38.0,-26.0},{38.0,-70.0},{66.0,-70.0}},
              color={255,0,255}),
            Line(points={{-80.0,0.0},{-62.0,0.0},{40.0,90.0},{40.0,0.0},{68.0,0.0}},
              color={0,0,127})}),
          Diagram(coordinateSystem(preserveAspectRatio=true, extent={{-100,-100},{
                  100,100}}), graphics={Line(points={{-90,-70},{82,-70}}, color={0,
                0,0}),Line(points={{-80,68},{-80,-80}}),Polygon(
                  points={{90,-70},{68,-62},{68,-78},{90,-70}},
                  lineColor={0,0,0},
                  fillColor={255,255,255},
                  fillPattern=FillPattern.Solid),Polygon(
                  points={{-80,90},{-88,68},{-72,68},{-80,90}},
                  lineColor={0,0,0},
                  fillColor={255,255,255},
                  fillPattern=FillPattern.Solid),Line(points={{-80,-68},{-60,-68},{
                -60,-40},{20,-40},{20,-68},{60,-68}}, color={255,0,255}),Line(
                points={{-80,-20},{-60,-20},{20,60},{20,-20},{60,-20},{60,-20}},
                color={0,0,255}),Text(
                  extent={{-88,6},{-54,-4}},
                  lineColor={0,0,0},
                  textString="y"),Text(
                  extent={{48,-80},{84,-88}},
                  lineColor={0,0,0},
                  textString="time"),Text(
                  extent={{-88,-36},{-54,-46}},
                  lineColor={0,0,0},
                  textString="u")}),
          Documentation(info="<html>
<p>When the Boolean input &quot;u&quot; becomes true, the timer is started and the output &quot;y&quot; is the time from the time instant where u became true. The timer is stopped and the output is reset to zero, once the input becomes false. </p>
</html>"));
      end Timer;

      model TimerGreat
        "Timer calculating the time when A is greater than B"

        parameter Real threshold=0 "Comparison with respect to threshold";

        Modelica.Blocks.Interfaces.RealInput u "Connector of Boolean input signal"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Interfaces.RealOutput y(
          final quantity="Time",
          final unit="s")
          "Connector of Real output signal"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));

        Modelica.Blocks.Logical.GreaterThreshold      greEqu(
           threshold = threshold)
          annotation (Placement(transformation(extent={{-50,-10},{-30,10}})));

        Timer tim "Timer"
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));

      equation
        connect(greEqu.y, tim.u)
          annotation (Line(points={{-29,0},{18,0}}, color={255,0,255}));
        connect(greEqu.u, u)
          annotation (Line(points={{-52,0},{-120,0}}, color={0,0,127}));
        connect(tim.y,y)  annotation (Line(points={{41,0},{110,0}}, color={0,0,127}));
        annotation (defaultComponentName="greEqu",
        Icon(coordinateSystem(preserveAspectRatio=false), graphics={
              Rectangle(
                extent={{-100,100},{100,-100}},
                lineColor={0,0,0},
                lineThickness=5.0,
                fillColor={210,210,210},
                fillPattern=FillPattern.Solid,
                borderPattern=BorderPattern.Raised),
                                         Text(
                extent={{-90,-40},{60,40}},
                lineColor={0,0,0},
                textString=">"),
              Ellipse(
                extent={{71,7},{85,-7}},
                lineColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillPattern=FillPattern.Solid),
                                              Text(
              extent={{-150,152},{150,112}},
              textString="%name",
              lineColor={0,0,255})}),                                  Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          Documentation(info="<html>
<p>This model represents a timer that starts to calculate the time when the input is greater than or equal to a certain threshold. It will return to zero when the condition does not satisfy.</p>
</html>"));
      end TimerGreat;

      model TimerGreatEqual
        "Timer calculating the time when A is greater than or equal than B"

        parameter Real threshold=0 "Comparison with respect to threshold";

        Modelica.Blocks.Interfaces.RealInput u "Connector of Boolean input signal"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Interfaces.RealOutput y(
          final quantity="Time",
          final unit="s")
          "Connector of Real output signal"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));

        Modelica.Blocks.Logical.GreaterEqualThreshold greEqu(
           threshold = threshold)
          annotation (Placement(transformation(extent={{-50,-10},{-30,10}})));

        Timer tim "Timer"
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));

      equation
        connect(greEqu.y, tim.u)
          annotation (Line(points={{-29,0},{18,0}}, color={255,0,255}));
        connect(greEqu.u, u)
          annotation (Line(points={{-52,0},{-120,0}}, color={0,0,127}));
        connect(tim.y,y)  annotation (Line(points={{41,0},{110,0}}, color={0,0,127}));
        annotation (defaultComponentName="greEqu",
        Icon(coordinateSystem(preserveAspectRatio=false), graphics={
              Rectangle(
                extent={{-100,100},{100,-100}},
                lineColor={0,0,0},
                lineThickness=5.0,
                fillColor={210,210,210},
                fillPattern=FillPattern.Solid,
                borderPattern=BorderPattern.Raised),
                                         Text(
                extent={{-90,-40},{60,40}},
                lineColor={0,0,0},
                textString=">="),
              Ellipse(
                extent={{71,7},{85,-7}},
                lineColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillPattern=FillPattern.Solid),
                                              Text(
              extent={{-150,152},{150,112}},
              textString="%name",
              lineColor={0,0,255})}),                                  Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          Documentation(info="<html>
<p>This model represents a timer that starts to calculate the time when the input is greater than or equal to a certain threshold. It will return to zero when the condition does not satisfy.</p>
</html>"));
      end TimerGreatEqual;

      model TimeLess "Timer calculating the time when A is less than B"

        parameter Real threshold=0 "Comparison with respect to threshold";

        Modelica.Blocks.Interfaces.RealInput u "Connector of Boolean input signal"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Interfaces.RealOutput y(
          final quantity="Time",
          final unit="s")
          "Connector of Real output signal"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));

        Modelica.Blocks.Logical.LessThreshold         lesEqu(
           threshold = threshold)
          annotation (Placement(transformation(extent={{-52,-10},{-32,10}})));

        Timer tim "Timer"
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));

      equation
        connect(lesEqu.y, tim.u)
          annotation (Line(points={{-31,0},{18,0}}, color={255,0,255}));
        connect(lesEqu.u, u)
          annotation (Line(points={{-54,0},{-120,0}}, color={0,0,127}));
        connect(tim.y,y)  annotation (Line(points={{41,0},{110,0}}, color={0,0,127}));
        annotation (defaultComponentName="lesEqu",
        Icon(coordinateSystem(preserveAspectRatio=false), graphics={
              Rectangle(
                extent={{-100,100},{100,-100}},
                lineColor={0,0,0},
                lineThickness=5.0,
                fillColor={210,210,210},
                fillPattern=FillPattern.Solid,
                borderPattern=BorderPattern.Raised),
                                         Text(
                extent={{-90,-40},{60,40}},
                lineColor={0,0,0},
                textString="<"),
              Ellipse(
                extent={{71,7},{85,-7}},
                lineColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillPattern=FillPattern.Solid),
                                              Text(
              extent={{-150,150},{150,110}},
              textString="%name",
              lineColor={0,0,255})}),                                  Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          Documentation(info="<html>
<p>This model represents a timer that starts to calculate the time when the input is less than a certain threshold. It will return to zero when the condition does not satisfy.</p>
</html>"));
      end TimeLess;

      model TimeLessEqual
        "Timer calculating the time when A is less than or equal than B"

        parameter Real threshold=0 "Comparison with respect to threshold";

        Modelica.Blocks.Interfaces.RealInput u "Connector of Boolean input signal"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));
        Modelica.Blocks.Interfaces.RealOutput y(
          final quantity="Time",
          final unit="s")
          "Connector of Real output signal"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));

        Modelica.Blocks.Logical.LessEqualThreshold    lesEqu(
           threshold = threshold)
          annotation (Placement(transformation(extent={{-52,-10},{-32,10}})));

        Timer tim "Timer"
          annotation (Placement(transformation(extent={{20,-10},{40,10}})));

      equation
        connect(lesEqu.y, tim.u)
          annotation (Line(points={{-31,0},{18,0}}, color={255,0,255}));
        connect(lesEqu.u, u)
          annotation (Line(points={{-54,0},{-120,0}}, color={0,0,127}));
        connect(tim.y,y)  annotation (Line(points={{41,0},{110,0}}, color={0,0,127}));
        annotation (defaultComponentName="lesEqu",
        Icon(coordinateSystem(preserveAspectRatio=false), graphics={
              Rectangle(
                extent={{-100,100},{100,-100}},
                lineColor={0,0,0},
                lineThickness=5.0,
                fillColor={210,210,210},
                fillPattern=FillPattern.Solid,
                borderPattern=BorderPattern.Raised),
                                         Text(
                extent={{-90,-40},{60,40}},
                lineColor={0,0,0},
                textString="<="),
              Ellipse(
                extent={{71,7},{85,-7}},
                lineColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillColor=DynamicSelect({235,235,235}, if y > 0.5 then {0,255,0}
                     else {235,235,235}),
                fillPattern=FillPattern.Solid),
                                              Text(
              extent={{-150,150},{150,110}},
              textString="%name",
              lineColor={0,0,255})}),                                  Diagram(
              coordinateSystem(preserveAspectRatio=false)),
          Documentation(info="<html>
<p>This model represents a timer that starts to calculate the time when the input is less than or equal to a certain threshold. It will return to zero when the condition does not satisfy.</p>
</html>"));
      end TimeLessEqual;

      block TrimAndRespond "Trim and respond logic"
        extends Modelica.Blocks.Interfaces.DiscreteSISO(firstTrigger(start=false, fixed=true));
        parameter Real uTri "Value to triggering the request for actuator";
        parameter Real yEqu0 "y setpoint when equipment starts";
        parameter Real yDec(max=0) "y decrement (must be negative)";
        parameter Real yInc(min=0) "y increment (must be positive)";

        Modelica.Blocks.Logical.GreaterEqualThreshold incY(threshold=uTri)
          "Outputs true if y needs to be increased"
          annotation (extent=[-20, 98; 0, 118], Placement(transformation(extent={{-20,
                  50},{0,70}})));
        Modelica.Blocks.Logical.Switch swi annotation (extent=[100, 110; 120, 130],
            Placement(transformation(extent={{60,50},{80,70}})));
        Sampler sam(samplePeriod=samplePeriod) "Sampler"
          annotation (extent=[-60, 90; -40, 110], Placement(transformation(extent={{-60,
                  50},{-40,70}})));

        Modelica.Blocks.Sources.Constant conYDec(k=yDec) "y decrease"
          annotation (extent=[26, 90; 46, 110], Placement(transformation(extent={{20,30},
                  {40,50}})));
        Modelica.Blocks.Sources.Constant conYInc(k=yInc) "y increase"
          annotation (extent=[-20, 124; 0, 144], Placement(transformation(extent={{20,70},
                  {40,90}})));
        UnitDelay uniDel1(
          y_start=yEqu0,
          samplePeriod=samplePeriod,
          startTime=samplePeriod)
                         annotation (extent=[-52, -40; -32, -20], Placement(
              transformation(extent={{-60,-16},{-40,4}})));
        Modelica.Blocks.Math.Add add annotation (extent=[-20, -20; 0, 0], Placement(
              transformation(extent={{-20,-10},{0,10}})));
        Modelica.Blocks.Nonlinear.Limiter lim(uMax=1, uMin=0) "State limiter"
          annotation (extent=[20, -20; 40, 0], Placement(transformation(extent={{20,-10},
                  {40,10}})));

        // The UnitDelay and Sampler is reimplemented to avoid in Dymola 2016 the translation warning
        //   The initial conditions for variables of type Boolean are not fully specified.
        //   Dymola has selected default initial conditions.
        //   Assuming fixed default start value for the discrete non-states:
        //     ...firstTrigger(start = false)
        //     ...

      protected
        block UnitDelay
          extends Modelica.Blocks.Discrete.UnitDelay(
            firstTrigger(start=false, fixed=true));
        end UnitDelay;

        block Sampler
          extends Modelica.Blocks.Discrete.Sampler(
            firstTrigger(start=false, fixed=true));
        end Sampler;
      equation
        connect(lim.y, y) annotation (Line(
            points={{41,6.10623e-16},{70,6.10623e-16},{70,5.55112e-16},{110,5.55112e-16}},
            color={0,0,127},
            smooth=Smooth.None));

        connect(add.y, lim.u) annotation (Line(
            points={{1,6.10623e-16},{9.5,6.10623e-16},{9.5,6.66134e-16},{18,
                6.66134e-16}},
            color={0,0,127},
            smooth=Smooth.None));

        connect(uniDel1.y, add.u2) annotation (Line(
            points={{-39,-6},{-22,-6}},
            color={0,0,127},
            smooth=Smooth.None));

        connect(incY.y, swi.u2) annotation (Line(
            points={{1,60},{58,60}},
            color={255,0,255},
            smooth=Smooth.None));
        connect(sam.y, incY.u) annotation (Line(
            points={{-39,60},{-22,60}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(lim.y, uniDel1.u) annotation (Line(
            points={{41,6.66134e-16},{60,6.66134e-16},{60,-40},{-80,-40},{-80,-6},{-62,
                -6}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(swi.y, add.u1) annotation (Line(
            points={{81,60},{88,60},{88,20},{-30,20},{-30,6},{-22,6}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(swi.u3, conYDec.y) annotation (Line(
            points={{58,52},{50,52},{50,40},{41,40}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(conYInc.y, swi.u1) annotation (Line(
            points={{41,80},{50,80},{50,68},{58,68}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(sam.u, u) annotation (Line(
            points={{-62,60},{-80,60},{-80,1.11022e-15},{-120,1.11022e-15}},
            color={0,0,127},
            smooth=Smooth.None));
        annotation (
          defaultComponentName="triAndRes",
          Documentation(info="<html>
<p>
   This model implements the trim and respond logic. The model samples the outputs of actuators every <code>tSam</code>.
   The control sequence is as follows:
</p>
<ul>
  <li>If <code>u &ge; 0</code>, then <code>y = y + nActInc</code>,</li>
  <li>If <code>u &lt; 0</code>, then <code>y = y - yDec</code>.</li>
</ul>
</html>",       revisions="<html>
<ul>
<li>
September 24, 2015 by Michael Wetter:<br/>
Implemented <code>UnitDelay</code> and <code>Sampler</code> to avoid a translation warning
because these blocks do not set the <code>fixed</code> attribute of <code>firstTrigger</code>
in MSL 3.2.1.
</li>
<li>
December 5, 2012, by Michael Wetter:<br/>
Simplified implementation.
</li>
<li>
September 21, 2012, by Wangda Zuo:<br/>
Deleted the status input that was not needed for new control.
</li>
<li>
July 20, 2011, by Wangda Zuo:<br/>
Added comments, redefine variable names, and merged to library.
</li>
<li>
January 6 2011, by Michael Wetter and Wangda Zuo:<br/>
First implementation.
</li>
</ul>
</html>"));
      end TrimAndRespond;

      block TrimAndRespondContinuousTimeApproximation
        "Trim and respond logic"
        extends Modelica.Blocks.Interfaces.SISO;
        parameter Real uTri "Value to triggering the request for actuator";
        parameter Real k=0.1 "Gain of controller";
        parameter Modelica.SIunits.Time Ti=120 "Time constant of Integrator block";

        Buildings.Controls.Continuous.LimPID conPID(
          Td=1,
          reverseAction=true,
          controllerType=Modelica.Blocks.Types.SimpleController.PI,
          k=k,
          Ti=Ti)     annotation (Placement(transformation(extent={{-20,40},{0,60}})));
        Modelica.Blocks.Sources.Constant const(k=uTri)
          annotation (Placement(transformation(extent={{-60,40},{-40,60}})));

      equation
        connect(const.y, conPID.u_s) annotation (Line(
            points={{-39,50},{-22,50}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(conPID.y, y) annotation (Line(
            points={{1,50},{76,50},{76,4.44089e-16},{110,4.44089e-16}},
            color={0,0,127},
            smooth=Smooth.None));
        connect(u, conPID.u_m) annotation (Line(
            points={{-120,0},{-10,0},{-10,38}},
            color={0,0,127},
            smooth=Smooth.None));
        annotation (
          defaultComponentName="triAndRes",
          Documentation(info="<html>
<p>
   This model implements a continuous time approximation to the trim and respond
   control algorithm.
</p>
   </html>",       revisions="<html>
<ul>
<li>
December 5, 2012, by Michael Wetter:<br/>
First implementation.
</li>
</ul>
</html>"));
      end TrimAndRespondContinuousTimeApproximation;

      package Validation "Collection of validation models that the base classes of the local controls"
        extends Modelica.Icons.ExamplesPackage;

        model Stage "Test the general stage model"
          extends Modelica.Icons.Example;
          parameter Integer numSta=4 "Design number of equipment that can be staged";
          parameter Real staUpThr=0.8 "Staging up threshold";
          parameter Real staDowThr=0.45 "Staging down threshold";
          FiveZone.PrimarySideControl.BaseClasses.Stage sta(
            staUpThr=staUpThr,
            staDowThr=staDowThr,
            waiTimStaUp=300,
            shoCycTim=200)
            annotation (Placement(transformation(extent={{-10,-10},{10,10}})));

          Buildings.Controls.OBC.CDL.Continuous.Sources.Sine u(
            amplitude=0.5,
            offset=0.5,
            freqHz=1/1500) "Input signal"
            annotation (Placement(transformation(extent={{-60,20},{-40,40}})));

          Modelica.Blocks.Sources.BooleanPulse on(period=3000)
            annotation (Placement(transformation(extent={{-60,-40},{-40,-20}})));
        equation
          connect(u.y, sta.u)
            annotation (Line(points={{-39,30},{-26,30},{-26,4},{-12,4}},
                                                       color={0,0,127}));
          connect(on.y, sta.on) annotation (Line(points={{-39,-30},{-26,-30},{-26,-4},{
                  -12,-4}}, color={255,0,255}));
          annotation (Icon(coordinateSystem(preserveAspectRatio=false)), Diagram(
                coordinateSystem(preserveAspectRatio=false)),
            experiment(StopTime=3600),
            __Dymola_Commands(file=
                  "modelica://WSEControlLogics/Resources/Scripts/Dymola/Controls/LocalControls/BaseClasses/Validation/Stage.mos"
                "Simulate and Plot"));
        end Stage;
        annotation (Documentation(info="<html>
<p>This package contains validation models for the classes in 
<a href=\"modelica://WSEControlLogics.Controls.LocalControls.BaseClasses\">
WSEControlLogics/Controls/LocalControls/BaseClasses</a>. </p>
</html>"));
      end Validation;
      annotation (Documentation(info="<html>
<p>This package contains base classes that are used to construct the models in 
<a href=\"modelica://WSEControlLogics.Controls.LocalControls\">
WSEControlLogics.Controls.LocalControl</a>.
</p>
</html>"));
    end BaseClasses;
  annotation (Documentation(info="<html>
<p>This package contains a collection of models for the local controls of chilled water system with waterside economizer. </p>
</html>"),   Icon(graphics={
        Rectangle(
          origin={0.0,35.1488},
          fillColor={255,255,255},
          extent={{-30.0,-20.1488},{30.0,20.1488}}),
        Polygon(
          origin={-40.0,35.0},
          pattern=LinePattern.None,
          fillPattern=FillPattern.Solid,
          points={{10.0,0.0},{-5.0,5.0},{-5.0,-5.0}}),
        Line(
          origin={-51.25,0.0},
          points={{21.25,-35.0},{-13.75,-35.0},{-13.75,35.0},{6.25,35.0}}),
        Line(
          origin={51.25,0.0},
          points={{-21.25,35.0},{13.75,35.0},{13.75,-35.0},{-6.25,-35.0}}),
        Rectangle(
          origin={0.0,-34.8512},
          fillColor={255,255,255},
          extent={{-30.0,-20.1488},{30.0,20.1488}})}));
  end PrimarySideControl;

  package Data "Performance data"

    record Chiller =
      Buildings.Fluid.Chillers.Data.ElectricEIR.Generic (
        QEva_flow_nominal =  -1076100,
        COP_nominal =         5.52,
        PLRMin =              0.10,
        PLRMinUnl =           0.10,
        PLRMax =              1.02,
        mEva_flow_nominal =   1000 * 0.03186,
        mCon_flow_nominal =   1000 * 0.04744,
        TEvaLvg_nominal =     273.15 + 5.56,
        TConEnt_nominal =     273.15 + 24.89,
        TEvaLvgMin =          273.15 + 5.56,
        TEvaLvgMax =          273.15 + 10.00,
        TConEntMin =          273.15 + 12.78,
        TConEntMax =          273.15 + 24.89,
        capFunT =             {1.785912E-01,-5.900023E-02,-5.946963E-04,9.297889E-02,-2.841024E-03,4.974221E-03},
        EIRFunT =             {5.245110E-01,-2.850126E-02,8.034720E-04,1.893133E-02,1.151629E-04,-9.340642E-05},
        EIRFunPLR =           {2.619878E-01,2.393605E-01,4.988306E-01},
        etaMotor =            1.0)
      "ElectricEIRChiller Carrier 19XR 1076kW/5.52COP/Vanes" annotation (
      defaultComponentName="datChi",
      defaultComponentPrefixes="parameter",
      Documentation(info=
                     "<html>
Performance data for chiller model.
This data corresponds to the following EnergyPlus model:

</html>"));
  end Data;

  package Types "Package with type definitions"
    extends Modelica.Icons.TypesPackage;

    type CoolingModes = enumeration(
        FreeCooling "Free cooling mode",
        PartialMechanical "Partial mechanical cooling",
        FullMechanical "Full mechanical cooling",
        Off "Off") annotation (
        Documentation(info="<html>
<p>Enumeration for the type cooling mode. </p>
<ol>
<li>FreeCooling </li>
<li>PartialMechanical </li>
<li>FullMechanical </li>
<li>Off</li>
</ol>
</html>",     revisions=
                      "<html>
<ul>
<li>
August 29, 2017, by Michael Wetter:<br/>
First implementation.
</li>
</ul>
</html>"));
    annotation (Documentation(info="<html>
<p>
This package contains type definitions.
</p>
</html>"));
  end Types;
annotation (uses(
    Buildings(version="7.0.0"),
    Modelica(version="3.2.3"),
      Complex(version="3.2.3"),
      Modelica_LinearSystems2(version="2.3.5"),
      Modelica_Synchronous(version="0.93.0")),
  version="1.0.0",
  conversion(noneFromVersion=""));
end FiveZone;
