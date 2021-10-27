within ;
package SingleZoneVAV
  "This package contains models for the SingleZoneVAV testcase in BOPTEST."

  model ZoneTemperature
    "Based on Buildings.Air.Systems.SingleZone.VAV.Examples.ChillerDXHeatingEconomizer."

    package MediumA = Buildings.Media.Air(extraPropertiesNames={"CO2"}) "Buildings library air media package";
    package MediumW = Buildings.Media.Water "Buildings library air media package";

    parameter Modelica.SIunits.Temperature TSupChi_nominal=279.15
      "Design value for chiller leaving water temperature";

    parameter Modelica.SIunits.Temperature THeaOn=293.15
      "Heating setpoint during on";
    parameter Modelica.SIunits.Temperature THeaOff=285.15
      "Heating setpoint during off";
    parameter Modelica.SIunits.Temperature TCooOn=297.15
      "Cooling setpoint during on";
    parameter Modelica.SIunits.Temperature TCooOff=303.15
      "Cooling setpoint during off";

    BaseClasses.Control.ChillerDXHeatingEconomizerController con(
      minAirFlo=0.1,
      minOAFra=0.15,
      kFan=4,
      kEco=4,
      kHea=4,
      TSupChi_nominal=TSupChi_nominal,
      TSetSupAir=286.15) "Controller"
      annotation (Placement(transformation(extent={{-100,-10},{-80,10}})));
    BaseClasses.ZoneTemperature hvac(
      redeclare package MediumA = MediumA,
      redeclare package MediumW = MediumW,
      mAir_flow_nominal=0.75,
      etaHea_nominal=0.99,
      QHea_flow_nominal=7000,
      QCoo_flow_nominal=-7000,
      TSupChi_nominal=TSupChi_nominal) "Single zone VAV system"
      annotation (Placement(transformation(extent={{-40,-20},{0,20}})));
    BaseClasses.Room  zon(
      redeclare package MediumA = MediumA,
        mAir_flow_nominal=0.75,
        lat=weaDat.lat,
      roo(mSenFac=4))   "Thermal envelope of single zone"
      annotation (Placement(transformation(extent={{40,-20},{80,20}})));
    Buildings.BoundaryConditions.WeatherData.ReaderTMY3 weaDat(
        computeWetBulbTemperature=false, filNam=
        Modelica.Utilities.Files.loadResource(
            "Resources/weatherdata/USA_CA_Riverside.Muni.AP.722869_TMY3.mos"))
      annotation (Placement(transformation(extent={{-160,120},{-140,140}})));

    Buildings.BoundaryConditions.WeatherData.Bus weaBus "Weather data bus"
      annotation (Placement(transformation(extent={{-118,120},{-98,140}})));

    IBPSA.Utilities.IO.SignalExchange.Overwrite
                             oveTSetRooHea(
                              u(
        unit="K",
        min=273.15 + 10,
        max=273.15 + 35), description="Heating setpoint")
      annotation (Placement(transformation(extent={{-140,20},{-120,40}})));
    IBPSA.Utilities.IO.SignalExchange.Overwrite
                             oveTSetRooCoo(
                              u(
        unit="K",
        min=273.15 + 10,
        max=273.15 + 35), description="Cooling setpoint")
      annotation (Placement(transformation(extent={{-140,-20},{-120,0}})));
    IBPSA.Utilities.IO.SignalExchange.Read
                        PPum(y(unit="W"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
      description="Pump electrical power")
      annotation (Placement(transformation(extent={{120,70},{140,90}})));
    IBPSA.Utilities.IO.SignalExchange.Read
                        PCoo(y(unit="W"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
      description="Cooling electrical power")
      annotation (Placement(transformation(extent={{140,90},{160,110}})));
    IBPSA.Utilities.IO.SignalExchange.Read
                        PHea(y(unit="W"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
      description="Heater power")
      annotation (Placement(transformation(extent={{120,110},{140,130}})));
    IBPSA.Utilities.IO.SignalExchange.Read
                        PFan(y(unit="W"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
      description="Fan electrical power")
      annotation (Placement(transformation(extent={{140,130},{160,150}})));
    IBPSA.Utilities.IO.SignalExchange.Read TRooAir(               y(unit="K"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.AirZoneTemperature,
      description="Room air temperature")
      annotation (Placement(transformation(extent={{120,-10},{140,10}})));
    IBPSA.Utilities.IO.SignalExchange.Read senTSetRooCoo(
      y(unit="K"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      description="Room cooling setpoint")
      annotation (Placement(transformation(extent={{-100,-80},{-80,-60}})));
    IBPSA.Utilities.IO.SignalExchange.Read senTSetRooHea(
      y(unit="K"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      description="Room heating setpoint")
      annotation (Placement(transformation(extent={{-100,40},{-80,60}})));
    IBPSA.Utilities.IO.SignalExchange.Read CO2RooAir(
      y(unit="ppm"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.CO2Concentration,
      description="Room air CO2 concentration") "CO2 concentration of room air"
      annotation (Placement(transformation(extent={{120,-40},{140,-20}})));

    Modelica.Blocks.Interfaces.RealOutput TRoo
      "Connector of Real output signal"
      annotation (Placement(transformation(extent={{160,-10},{180,10}})));
    Modelica.Blocks.Interfaces.RealOutput CO2Roo
      "Connector of Real output signal"
      annotation (Placement(transformation(extent={{160,-40},{180,-20}})));
    Modelica.Blocks.Math.MultiSum PHVAC(nu=3)
      annotation (Placement(transformation(extent={{126,34},{138,46}})));
    Modelica.Blocks.Interfaces.RealOutput PTot
      annotation (Placement(transformation(extent={{160,30},{180,50}})));
    Modelica.Blocks.Interfaces.RealOutput TOut
      annotation (Placement(transformation(extent={{160,-70},{180,-50}})));
    Modelica.Blocks.Interfaces.RealOutput GHI
      annotation (Placement(transformation(extent={{160,-90},{180,-70}})));
    Modelica.Blocks.Sources.CombiTimeTable TSetRooHea(
      smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
      extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
      table=[0,THeaOff; 7*3600,THeaOff; 7*3600,THeaOn; 19*3600,THeaOn; 19*3600,
          THeaOff; 24*3600,THeaOff]) "Heating setpoint for room temperature"
      annotation (Placement(transformation(extent={{-180,20},{-160,40}})));
    Modelica.Blocks.Interfaces.RealInput TSetCoo
      "Connector of Real input signal"
      annotation (Placement(transformation(extent={{-200,-60},{-160,-20}})));
    Buildings.Controls.SetPoints.OccupancySchedule occSch
      "Occupancy schedule"
      annotation (Placement(transformation(extent={{-140,80},{-120,100}})));
    Modelica.Blocks.Sources.Constant zer(k=0)
      annotation (Placement(transformation(extent={{-100,-120},{-80,-100}})));
  equation
    connect(weaDat.weaBus, weaBus) annotation (Line(
        points={{-140,130},{-108,130}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%second",
        index=1,
        extent={{6,3},{6,3}}));

    connect(con.yCooCoiVal, hvac.uCooVal) annotation (Line(points={{-79,0},{-54,0},
            {-54,5},{-42,5}},             color={0,0,127}));
    connect(con.yOutAirFra, hvac.uEco) annotation (Line(points={{-79,3},{-50,3},{
            -50,-2},{-42,-2}},             color={0,0,127}));
    connect(con.TSetSupChi, hvac.TSetChi) annotation (Line(points={{-79,-8},{-70,
            -8},{-70,-15},{-42,-15}},           color={0,0,127}));
    connect(con.TMix, hvac.TMix) annotation (Line(points={{-102,0},{-112,0},{-112,
            -40},{10,-40},{10,-4},{1,-4}},                  color={0,0,127}));

    connect(hvac.supplyAir, zon.supplyAir) annotation (Line(points={{0,8},{10,8},
            {10,2},{40,2}},          color={0,127,255}));
    connect(hvac.returnAir, zon.returnAir) annotation (Line(points={{0,0},{6,0},{
            6,-2},{10,-2},{40,-2}},  color={0,127,255}));

    connect(con.TOut, weaBus.TDryBul) annotation (Line(points={{-102,-3},{-108,-3},
            {-108,130}},                  color={0,0,127}));
    connect(hvac.weaBus, weaBus) annotation (Line(
        points={{-36,17.8},{-36,130},{-108,130}},
        color={255,204,51},
        thickness=0.5));
    connect(zon.weaBus, weaBus) annotation (Line(
        points={{46,18},{42,18},{42,130},{-108,130}},
        color={255,204,51},
        thickness=0.5));
    connect(con.TSup, hvac.TSup) annotation (Line(points={{-102,-9},{-108,-9},{-108,
            -32},{4,-32},{4,-8},{1,-8}},
          color={0,0,127}));
    connect(con.TRoo, zon.TRooAir) annotation (Line(points={{-102,-6},{-110,-6},{
            -110,-36},{6,-36},{6,-22},{90,-22},{90,0},{81,0}},      color={0,0,
            127}));

    connect(oveTSetRooHea.y, con.TSetRooHea) annotation (Line(points={{-119,30},{-116,
            30},{-116,6},{-102,6}},         color={0,0,127}));
    connect(oveTSetRooCoo.y, con.TSetRooCoo) annotation (Line(points={{-119,-10},{
            -116,-10},{-116,3},{-102,3}},  color={0,0,127}));
    connect(hvac.PPum, PPum.u) annotation (Line(points={{1,12},{18,12},{18,80},
            {118,80}}, color={0,0,127}));
    connect(hvac.PCoo, PCoo.u) annotation (Line(points={{1,14},{14,14},{14,100},
            {138,100}}, color={0,0,127}));
    connect(hvac.QHea_flow, PHea.u) annotation (Line(points={{1,16},{10,16},{10,
            120},{118,120}}, color={0,0,127}));
    connect(hvac.PFan, PFan.u) annotation (Line(points={{1,18},{6,18},{6,140},{
            138,140}}, color={0,0,127}));
    connect(zon.TRooAir, TRooAir.u)
      annotation (Line(points={{81,0},{118,0}}, color={0,0,127}));
    connect(oveTSetRooCoo.y, senTSetRooCoo.u) annotation (Line(points={{-119,
            -10},{-116,-10},{-116,-70},{-102,-70}}, color={0,0,127}));
    connect(oveTSetRooHea.y, senTSetRooHea.u) annotation (Line(points={{-119,30},
            {-116,30},{-116,50},{-102,50}}, color={0,0,127}));
    connect(zon.CO2, CO2RooAir.u) annotation (Line(points={{81,-4},{100,-4},{
            100,-30},{118,-30}},
                        color={0,0,127}));
    connect(TRooAir.y, TRoo)
      annotation (Line(points={{141,0},{170,0}}, color={0,0,127}));
    connect(CO2RooAir.y, CO2Roo)
      annotation (Line(points={{141,-30},{170,-30}}, color={0,0,127}));
    connect(PFan.y, PHVAC.u[1]) annotation (Line(points={{161,140},{174,140},{
            174,64},{94,64},{94,42.8},{126,42.8}},
                                                 color={0,0,127}));
    connect(PCoo.y, PHVAC.u[2]) annotation (Line(points={{161,100},{168,100},{
            168,68},{90,68},{90,40},{126,40}},   color={0,0,127}));
    connect(PPum.y, PHVAC.u[3]) annotation (Line(points={{141,80},{160,80},{160,
            68},{88,68},{88,37.2},{126,37.2}},
                                             color={0,0,127}));
    connect(PHVAC.y, PTot)
      annotation (Line(points={{139.02,40},{170,40}}, color={0,0,127}));
    connect(weaBus.TDryBul, TOut) annotation (Line(
        points={{-108,130},{112,130},{112,-60},{170,-60}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(weaBus.HGloHor, GHI) annotation (Line(
        points={{-108,130},{112,130},{112,-80},{170,-80}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(TSetRooHea.y[1], oveTSetRooHea.u)
      annotation (Line(points={{-159,30},{-142,30}}, color={0,0,127}));
    connect(con.yFan, hvac.uFan) annotation (Line(points={{-79,9},{-60,9},{-60,
            18},{-42,18}}, color={0,0,127}));
    connect(con.chiOn, hvac.chiOn) annotation (Line(points={{-79,-4},{-54,-4},{
            -54,-10},{-42,-10}}, color={255,0,255}));
    connect(TSetCoo, oveTSetRooCoo.u) annotation (Line(points={{-180,-40},{-150,
            -40},{-150,-10},{-142,-10}}, color={0,0,127}));
    connect(occSch.occupied, con.uOcc) annotation (Line(points={{-119,84},{-106,
            84},{-106,9},{-104,9}}, color={255,0,255}));
    connect(zer.y, hvac.uHea) annotation (Line(points={{-79,-110},{-56,-110},{
            -56,12},{-42,12}}, color={0,0,127}));
    annotation (
      experiment(
        StartTime=18316800,
        StopTime=20908800,
        __Dymola_Algorithm="Cvode"),
        __Dymola_Commands(file="modelica://Buildings/Resources/Scripts/Dymola/Air/Systems/SingleZone/VAV/Examples/ChillerDXHeatingEconomizer.mos"
          "Simulate and plot"),
       Documentation(info="<html>
<p>
The thermal zone is based on the BESTEST Case 600 envelope, while the HVAC
system is based on a conventional VAV system with air cooled chiller and
economizer.  See documentation for the specific models for more information.
</p>
</html>",   revisions="<html>
<ul>
<li>
September 14, 2018, by David Blum:<br/>
First implementation.
</li>
</ul>
</html>"),
      Diagram(coordinateSystem(extent={{-160,-160},{120,140}})),
      Icon(coordinateSystem(extent={{-160,-160},{120,140}}), graphics={
          Rectangle(
            extent={{-160,140},{120,-160}},
            lineColor={0,0,0},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid),
          Polygon(lineColor = {0,0,255},
                  fillColor = {75,138,73},
                  pattern = LinePattern.None,
                  fillPattern = FillPattern.Solid,
                  points = {{-36,60},{64,0},{-36,-60},{-36,60}}),
          Ellipse(lineColor = {75,138,73},
                  fillColor={255,255,255},
                  fillPattern = FillPattern.Solid,
                  extent={{-116,-110},{84,90}}),
          Polygon(lineColor = {0,0,255},
                  fillColor = {75,138,73},
                  pattern = LinePattern.None,
                  fillPattern = FillPattern.Solid,
                  points={{-52,50},{48,-10},{-52,-70},{-52,50}})}));
  end ZoneTemperature;

  model ZoneTemperatureBaseline
    "Based on Buildings.Air.Systems.SingleZone.VAV.Examples.ChillerDXHeatingEconomizer."

    package MediumA = Buildings.Media.Air(extraPropertiesNames={"CO2"}) "Buildings library air media package";
    package MediumW = Buildings.Media.Water "Buildings library air media package";

    parameter Modelica.SIunits.Temperature TSupChi_nominal=279.15
      "Design value for chiller leaving water temperature";

    parameter Modelica.SIunits.Temperature THeaOn=293.15
      "Heating setpoint during on";
    parameter Modelica.SIunits.Temperature THeaOff=285.15
      "Heating setpoint during off";
    parameter Modelica.SIunits.Temperature TCooOn=297.15
      "Cooling setpoint during on";
    parameter Modelica.SIunits.Temperature TCooOff=303.15
      "Cooling setpoint during off";

    BaseClasses.Control.ChillerDXHeatingEconomizerController con(
      minAirFlo=0.1,
      minOAFra=0.15,
      kCoo=0.5,
      kFan=4,
      kEco=4,
      kHea=4,
      TSupChi_nominal=TSupChi_nominal,
      TSetSupAir=286.15) "Controller"
      annotation (Placement(transformation(extent={{-100,-10},{-80,10}})));
    BaseClasses.ZoneTemperature hvac(
      redeclare package MediumA = MediumA,
      redeclare package MediumW = MediumW,
      mAir_flow_nominal=0.75,
      etaHea_nominal=0.99,
      QHea_flow_nominal=7000,
      QCoo_flow_nominal=-7000,
      TSupChi_nominal=TSupChi_nominal) "Single zone VAV system"
      annotation (Placement(transformation(extent={{-40,-20},{0,20}})));
    BaseClasses.Room  zon(
      redeclare package MediumA = MediumA,
        mAir_flow_nominal=0.75,
        lat=weaDat.lat,
      roo(mSenFac=4))   "Thermal envelope of single zone"
      annotation (Placement(transformation(extent={{40,-20},{80,20}})));
    Buildings.BoundaryConditions.WeatherData.ReaderTMY3 weaDat(
        computeWetBulbTemperature=false, filNam=
          ModelicaServices.ExternalReferences.loadResource(
          "modelica://Buildings/Resources/weatherdata/DRYCOLD.mos"))
      annotation (Placement(transformation(extent={{-160,120},{-140,140}})));

    Buildings.BoundaryConditions.WeatherData.Bus weaBus "Weather data bus"
      annotation (Placement(transformation(extent={{-118,120},{-98,140}})));

    IBPSA.Utilities.IO.SignalExchange.Overwrite
                             oveTSetRooHea(
                              u(
        unit="K",
        min=273.15 + 10,
        max=273.15 + 35), description="Heating setpoint")
      annotation (Placement(transformation(extent={{-140,20},{-120,40}})));
    IBPSA.Utilities.IO.SignalExchange.Overwrite
                             oveTSetRooCoo(
                              u(
        unit="K",
        min=273.15 + 10,
        max=273.15 + 35), description="Cooling setpoint")
      annotation (Placement(transformation(extent={{-140,-20},{-120,0}})));
    IBPSA.Utilities.IO.SignalExchange.Read
                        PPum(y(unit="W"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
      description="Pump electrical power")
      annotation (Placement(transformation(extent={{120,70},{140,90}})));
    IBPSA.Utilities.IO.SignalExchange.Read
                        PCoo(y(unit="W"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
      description="Cooling electrical power")
      annotation (Placement(transformation(extent={{140,90},{160,110}})));
    IBPSA.Utilities.IO.SignalExchange.Read
                        PHea(y(unit="W"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
      description="Heater power")
      annotation (Placement(transformation(extent={{120,110},{140,130}})));
    IBPSA.Utilities.IO.SignalExchange.Read
                        PFan(y(unit="W"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.ElectricPower,
      description="Fan electrical power")
      annotation (Placement(transformation(extent={{140,130},{160,150}})));
    IBPSA.Utilities.IO.SignalExchange.Read TRooAir(               y(unit="K"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.AirZoneTemperature,
      description="Room air temperature")
      annotation (Placement(transformation(extent={{120,-10},{140,10}})));
    IBPSA.Utilities.IO.SignalExchange.Read senTSetRooCoo(
      y(unit="K"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      description="Room cooling setpoint")
      annotation (Placement(transformation(extent={{-100,-80},{-80,-60}})));
    IBPSA.Utilities.IO.SignalExchange.Read senTSetRooHea(
      y(unit="K"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.None,
      description="Room heating setpoint")
      annotation (Placement(transformation(extent={{-100,40},{-80,60}})));
    IBPSA.Utilities.IO.SignalExchange.Read CO2RooAir(
      y(unit="ppm"),
      KPIs=IBPSA.Utilities.IO.SignalExchange.SignalTypes.SignalsForKPIs.CO2Concentration,
      description="Room air CO2 concentration") "CO2 concentration of room air"
      annotation (Placement(transformation(extent={{120,-40},{140,-20}})));

    Modelica.Blocks.Interfaces.RealOutput TRoo(
       final unit="K",
       displayUnit="degC")
      "Connector of Real output signal"
      annotation (Placement(transformation(extent={{160,-10},{180,10}})));
    Modelica.Blocks.Interfaces.RealOutput CO2Roo
      "Connector of Real output signal"
      annotation (Placement(transformation(extent={{160,-40},{180,-20}})));
    Modelica.Blocks.Math.MultiSum PHVAC(nu=4)
      annotation (Placement(transformation(extent={{126,34},{138,46}})));
    Modelica.Blocks.Interfaces.RealOutput PTot
      annotation (Placement(transformation(extent={{160,30},{180,50}})));
    Modelica.Blocks.Interfaces.RealOutput TOut(
      final unit="K",
      displayUnit="degC")
      annotation (Placement(transformation(extent={{160,-70},{180,-50}})));
    Modelica.Blocks.Interfaces.RealOutput GHI
      annotation (Placement(transformation(extent={{160,-90},{180,-70}})));
    Modelica.Blocks.Sources.CombiTimeTable TSetRooHea(
      smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
      extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
      table=[0,THeaOff; 7*3600,THeaOff; 7*3600,THeaOn; 19*3600,THeaOn; 19*3600,
          THeaOff; 24*3600,THeaOff]) "Heating setpoint for room temperature"
      annotation (Placement(transformation(extent={{-180,20},{-160,40}})));
    Modelica.Blocks.Sources.CombiTimeTable TSetRooCoo(
      smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
      extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
      table=[0,TCooOff; 7*3600,TCooOff; 7*3600,TCooOn; 19*3600,TCooOn; 19*3600,
          TCooOff; 24*3600,TCooOff]) "Cooling setpoint for room temperature"
      annotation (Placement(transformation(extent={{-180,-20},{-160,0}})));
    Buildings.Controls.SetPoints.OccupancySchedule occSch
      "Occupancy schedule"
      annotation (Placement(transformation(extent={{-140,80},{-120,100}})));
    Modelica.Blocks.Sources.Constant zer(k=0)
      annotation (Placement(transformation(extent={{-100,-120},{-80,-100}})));
  equation
    connect(weaDat.weaBus, weaBus) annotation (Line(
        points={{-140,130},{-108,130}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%second",
        index=1,
        extent={{6,3},{6,3}}));

    connect(con.yCooCoiVal, hvac.uCooVal) annotation (Line(points={{-79,0},{-54,0},
            {-54,5},{-42,5}},             color={0,0,127}));
    connect(con.yOutAirFra, hvac.uEco) annotation (Line(points={{-79,3},{-50,3},{
            -50,-2},{-42,-2}},             color={0,0,127}));
    connect(con.TSetSupChi, hvac.TSetChi) annotation (Line(points={{-79,-8},{-70,
            -8},{-70,-15},{-42,-15}},           color={0,0,127}));
    connect(con.TMix, hvac.TMix) annotation (Line(points={{-102,0},{-112,0},{-112,
            -40},{10,-40},{10,-4},{1,-4}},                  color={0,0,127}));

    connect(hvac.supplyAir, zon.supplyAir) annotation (Line(points={{0,8},{10,8},
            {10,2},{40,2}},          color={0,127,255}));
    connect(hvac.returnAir, zon.returnAir) annotation (Line(points={{0,0},{6,0},{
            6,-2},{10,-2},{40,-2}},  color={0,127,255}));

    connect(con.TOut, weaBus.TDryBul) annotation (Line(points={{-102,-3},{-108,-3},
            {-108,130}},                  color={0,0,127}));
    connect(hvac.weaBus, weaBus) annotation (Line(
        points={{-36,17.8},{-36,130},{-108,130}},
        color={255,204,51},
        thickness=0.5));
    connect(zon.weaBus, weaBus) annotation (Line(
        points={{46,18},{42,18},{42,130},{-108,130}},
        color={255,204,51},
        thickness=0.5));
    connect(con.TSup, hvac.TSup) annotation (Line(points={{-102,-9},{-108,-9},{-108,
            -32},{4,-32},{4,-8},{1,-8}},
          color={0,0,127}));
    connect(con.TRoo, zon.TRooAir) annotation (Line(points={{-102,-6},{-110,-6},{
            -110,-36},{6,-36},{6,-22},{90,-22},{90,0},{81,0}},      color={0,0,
            127}));

    connect(oveTSetRooHea.y, con.TSetRooHea) annotation (Line(points={{-119,30},{-116,
            30},{-116,6},{-102,6}},         color={0,0,127}));
    connect(oveTSetRooCoo.y, con.TSetRooCoo) annotation (Line(points={{-119,-10},{
            -116,-10},{-116,3},{-102,3}},  color={0,0,127}));
    connect(hvac.PPum, PPum.u) annotation (Line(points={{1,12},{18,12},{18,80},
            {118,80}}, color={0,0,127}));
    connect(hvac.PCoo, PCoo.u) annotation (Line(points={{1,14},{14,14},{14,100},
            {138,100}}, color={0,0,127}));
    connect(hvac.QHea_flow, PHea.u) annotation (Line(points={{1,16},{10,16},{10,
            120},{118,120}}, color={0,0,127}));
    connect(hvac.PFan, PFan.u) annotation (Line(points={{1,18},{6,18},{6,140},{
            138,140}}, color={0,0,127}));
    connect(zon.TRooAir, TRooAir.u)
      annotation (Line(points={{81,0},{118,0}}, color={0,0,127}));
    connect(oveTSetRooCoo.y, senTSetRooCoo.u) annotation (Line(points={{-119,
            -10},{-116,-10},{-116,-70},{-102,-70}}, color={0,0,127}));
    connect(oveTSetRooHea.y, senTSetRooHea.u) annotation (Line(points={{-119,30},
            {-116,30},{-116,50},{-102,50}}, color={0,0,127}));
    connect(zon.CO2, CO2RooAir.u) annotation (Line(points={{81,-4},{100,-4},{
            100,-30},{118,-30}},
                        color={0,0,127}));
    connect(TRooAir.y, TRoo)
      annotation (Line(points={{141,0},{170,0}}, color={0,0,127}));
    connect(CO2RooAir.y, CO2Roo)
      annotation (Line(points={{141,-30},{170,-30}}, color={0,0,127}));
    connect(PFan.y, PHVAC.u[1]) annotation (Line(points={{161,140},{174,140},{174,
            64},{94,64},{94,43.15},{126,43.15}}, color={0,0,127}));
    connect(PCoo.y, PHVAC.u[2]) annotation (Line(points={{161,100},{168,100},{168,
            68},{90,68},{90,41.05},{126,41.05}}, color={0,0,127}));
    connect(PPum.y, PHVAC.u[3]) annotation (Line(points={{141,80},{160,80},{160,68},
            {88,68},{88,38.95},{126,38.95}}, color={0,0,127}));
    connect(PHVAC.y, PTot)
      annotation (Line(points={{139.02,40},{170,40}}, color={0,0,127}));
    connect(weaBus.TDryBul, TOut) annotation (Line(
        points={{-108,130},{112,130},{112,-60},{170,-60}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(weaBus.HGloHor, GHI) annotation (Line(
        points={{-108,130},{112,130},{112,-80},{170,-80}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%first",
        index=-1,
        extent={{-6,3},{-6,3}},
        horizontalAlignment=TextAlignment.Right));
    connect(TSetRooHea.y[1], oveTSetRooHea.u)
      annotation (Line(points={{-159,30},{-142,30}}, color={0,0,127}));
    connect(con.yFan, hvac.uFan) annotation (Line(points={{-79,9},{-60,9},{-60,
            18},{-42,18}}, color={0,0,127}));
    connect(con.chiOn, hvac.chiOn) annotation (Line(points={{-79,-4},{-54,-4},{
            -54,-10},{-42,-10}}, color={255,0,255}));
    connect(TSetRooCoo.y[1], oveTSetRooCoo.u) annotation (Line(points={{-159,
            -10},{-149.5,-10},{-149.5,-10},{-142,-10}}, color={0,0,127}));
    connect(occSch.occupied, con.uOcc) annotation (Line(points={{-119,84},{-110,
            84},{-110,9},{-104,9}}, color={255,0,255}));
    connect(PHea.y, PHVAC.u[4]) annotation (Line(points={{141,120},{170,120},{170,
            60},{98,60},{98,36.85},{126,36.85}}, color={0,0,127}));
    connect(zer.y, hvac.uHea) annotation (Line(points={{-79,-110},{-58,-110},{
            -58,12},{-42,12}}, color={0,0,127}));
    annotation (
      experiment(
        StartTime=18316800,
        StopTime=20908800,
        Tolerance=1e-06,
        __Dymola_Algorithm="Cvode"),
        __Dymola_Commands(file="modelica://Buildings/Resources/Scripts/Dymola/Air/Systems/SingleZone/VAV/Examples/ChillerDXHeatingEconomizer.mos"
          "Simulate and plot"),
       Documentation(info="<html>
<p>
The thermal zone is based on the BESTEST Case 600 envelope, while the HVAC
system is based on a conventional VAV system with air cooled chiller and
economizer.  See documentation for the specific models for more information.
</p>
</html>",   revisions="<html>
<ul>
<li>
September 14, 2018, by David Blum:<br/>
First implementation.
</li>
</ul>
</html>"),
      Diagram(coordinateSystem(extent={{-160,-160},{120,140}})),
      Icon(coordinateSystem(extent={{-160,-160},{120,140}}), graphics={
          Rectangle(
            extent={{-160,140},{120,-160}},
            lineColor={0,0,0},
            fillColor={255,255,255},
            fillPattern=FillPattern.Solid),
          Polygon(lineColor = {0,0,255},
                  fillColor = {75,138,73},
                  pattern = LinePattern.None,
                  fillPattern = FillPattern.Solid,
                  points = {{-36,60},{64,0},{-36,-60},{-36,60}}),
          Ellipse(lineColor = {75,138,73},
                  fillColor={255,255,255},
                  fillPattern = FillPattern.Solid,
                  extent={{-116,-110},{84,90}}),
          Polygon(lineColor = {0,0,255},
                  fillColor = {75,138,73},
                  pattern = LinePattern.None,
                  fillPattern = FillPattern.Solid,
                  points={{-52,50},{48,-10},{-52,-70},{-52,50}})}));
  end ZoneTemperatureBaseline;

  package BaseClasses "Base classes for test case"
    extends Modelica.Icons.BasesPackage;
    model Room "Room model for test case"
      extends Buildings.Air.Systems.SingleZone.VAV.Examples.BaseClasses.Room(
        roo(use_C_flow=true, nPorts=6),
        sinInf(use_C_in=true),
        TRooAir(unit="K"));
      Modelica.Blocks.Math.Gain gaiCO2Gen(k=2)
        "Number of people for CO2 generation"
        annotation (Placement(transformation(extent={{-80,20},{-60,40}})));
      Buildings.Fluid.Sensors.Conversions.To_VolumeFraction conMasVolFra(MMMea=
            Modelica.Media.IdealGases.Common.SingleGasesData.CO2.MM)
        "Conversion from mass fraction CO2 to volume fraction CO2"
        annotation (Placement(transformation(extent={{100,-40},{120,-20}})));
      Buildings.Fluid.Sensors.TraceSubstances        senCO2(redeclare package
          Medium = MediumA)
                        "CO2 sensor"
        annotation (Placement(transformation(extent={{20,-60},{40,-40}})));
      Modelica.Blocks.Sources.Constant conCO2Out(k=400e-6*Modelica.Media.IdealGases.Common.SingleGasesData.CO2.MM
            /Modelica.Media.IdealGases.Common.SingleGasesData.Air.MM)
        "Outside air CO2 concentration"
        annotation (Placement(transformation(extent={{-120,-70},{-100,-50}})));
      Modelica.Blocks.Math.Gain gaiPpm(k=1e6) "Gain for CO2 generation in ppm"
        annotation (Placement(transformation(extent={{140,-40},{160,-20}})));
      Modelica.Blocks.Interfaces.RealOutput CO2(unit="ppm") "Room air CO2 concentration"
        annotation (Placement(transformation(extent={{200,-50},{220,-30}}),
            iconTransformation(extent={{200,-50},{220,-30}})));
      Modelica.Blocks.Sources.Constant mCO2Gai_flow(k=2.5*8.64e-6)
        "CO2 generation per person in kg/s (elevated by 2.5x to force ppm above limit for testing)"
        annotation (Placement(transformation(extent={{-180,80},{-160,100}})));
    protected
      Modelica.Blocks.Math.Product pro4 "Product for internal gain"
        annotation (Placement(transformation(extent={{-40,20},{-20,40}})));
    equation
      connect(conCO2Out.y, sinInf.C_in[1]) annotation (Line(points={{-99,-60},{-70,-60},
              {-70,-18},{-40,-18}}, color={0,0,127}));
      connect(senCO2.port, roo.ports[6]) annotation (Line(points={{30,-60},{30,-66},
              {20,-66},{20,-13},{40.5,-13}}, color={0,127,255}));
      connect(senCO2.C, conMasVolFra.m) annotation (Line(points={{41,-50},{50,-50},{
              50,-30},{99,-30}}, color={0,0,127}));
      connect(conMasVolFra.V, gaiPpm.u)
        annotation (Line(points={{121,-30},{138,-30}}, color={0,0,127}));
      connect(gaiPpm.y, CO2) annotation (Line(points={{161,-30},{180,-30},{180,-40},
              {210,-40}}, color={0,0,127}));
      connect(mCO2Gai_flow.y, gaiCO2Gen.u) annotation (Line(points={{-159,90},{-132,
              90},{-132,30},{-82,30}}, color={0,0,127}));
      connect(gaiCO2Gen.y, pro4.u2) annotation (Line(points={{-59,30},{-50,30},{-50,
              24},{-42,24}}, color={0,0,127}));
      connect(intLoad.y[1], pro4.u1) annotation (Line(points={{-99,160},{-94,160},{-94,
              158},{-90,158},{-90,56},{-48,56},{-48,36},{-42,36}}, color={0,0,127}));
      connect(pro4.y, roo.C_flow[1]) annotation (Line(points={{-19,30},{10,30},{10,3.64},
              {31.92,3.64}}, color={0,0,127}));
    end Room;

    model ZoneTemperature "RTU model for test case"
      extends SingleZoneVAV.BaseClasses.ZoneTemperatureBase(
          out(use_C_in=true));
      Modelica.Blocks.Sources.Constant conCO2Out(k=400e-6*Modelica.Media.IdealGases.Common.SingleGasesData.CO2.MM
            /Modelica.Media.IdealGases.Common.SingleGasesData.Air.MM)
        "Outside air CO2 concentration"
        annotation (Placement(transformation(extent={{-182,100},{-162,120}})));
    equation
      connect(conCO2Out.y, out.C_in[1]) annotation (Line(points={{-161,110},{-144,110},
              {-144,32},{-142,32}}, color={0,0,127}));
    end ZoneTemperature;

    model ZoneTemperatureBase
      "HVAC system model with a dry cooling coil, air-cooled chiller, electric heating coil, variable speed fan, and mixing box with economizer control."
      replaceable package MediumA = Buildings.Media.Air "Medium model for air"
          annotation (choicesAllMatching = true);
      replaceable package MediumW = Buildings.Media.Water "Medium model for water"
          annotation (choicesAllMatching = true);

      parameter Modelica.SIunits.DimensionlessRatio COP_nominal = 4
        "Nominal COP of the chiller";

      parameter Modelica.SIunits.Temperature TSupChi_nominal
        "Design value for chiller leaving water temperature";

      parameter Modelica.SIunits.MassFlowRate mAir_flow_nominal "Design airflow rate of system"
        annotation(Dialog(group="Air design"));

      parameter Modelica.SIunits.Power QHea_flow_nominal(min=0) "Design heating capacity of heating coil"
        annotation(Dialog(group="Heating design"));

      parameter Real etaHea_nominal(min=0, max=1, unit="1") "Design heating efficiency of the heating coil"
        annotation(Dialog(group="Heating design"));

      parameter Modelica.SIunits.Power QCoo_flow_nominal(max=0) "Design heating capacity of cooling coil"
        annotation(Dialog(group="Cooling design"));

      parameter Modelica.SIunits.PressureDifference dp_nominal(displayUnit="Pa") = 500
        "Design pressure drop of flow leg with fan"
        annotation(Dialog(group="Air design"));

      final parameter Modelica.SIunits.MassFlowRate mChiEva_flow_nominal=
        -QCoo_flow_nominal/Buildings.Utilities.Psychrometrics.Constants.cpWatLiq/4
        "Design chilled water supply flow";

      final parameter Modelica.SIunits.MassFlowRate mChiCon_flow_nominal=
        -QCoo_flow_nominal*(1+1/COP_nominal)/Buildings.Utilities.Psychrometrics.Constants.cpAir/10
        "Design condenser air flow";

      Modelica.Blocks.Interfaces.BooleanInput chiOn "On signal for chiller plant"
        annotation (Placement(transformation(extent={{-240,-160},{-200,-120}})));

      Modelica.Blocks.Interfaces.RealInput uFan(
        final unit="1") "Fan control signal"
        annotation (Placement(transformation(extent={{-240,120},{-200,160}})));
      Modelica.Blocks.Interfaces.RealInput uHea(
        final unit="1") "Control input for heater"
        annotation (Placement(transformation(extent={{-240,60},{-200,100}})));
      Modelica.Blocks.Interfaces.RealInput uCooVal(final unit="1")
        "Control signal for cooling valve"
        annotation (Placement(transformation(extent={{-240,-10},{-200,30}})));
      Modelica.Blocks.Interfaces.RealInput TSetChi(
        final unit="K",
        displayUnit="degC")
        "Set point for leaving chilled water temperature"
        annotation (Placement(transformation(extent={{-240,-210},{-200,-170}})));
      Modelica.Blocks.Interfaces.RealInput uEco "Control signal for economizer"
        annotation (Placement(transformation(extent={{-240,-80},{-200,-40}})));

      Modelica.Fluid.Interfaces.FluidPort_a supplyAir(
        redeclare final package Medium = MediumA) "Supply air"
        annotation (Placement(transformation(extent={{190,30},{210,50}}),
            iconTransformation(extent={{190,30},{210,50}})));
      Modelica.Fluid.Interfaces.FluidPort_b returnAir(
        redeclare final package Medium = MediumA) "Return air"
        annotation (Placement(transformation(extent={{190,-50},{210,-30}}),
            iconTransformation(extent={{190,-50},{210,-30}})));

      Modelica.Blocks.Interfaces.RealOutput PFan(final unit="W")
        "Electrical power consumed by the supply fan"
        annotation (Placement(transformation(extent={{200,130},{220,150}}),
            iconTransformation(extent={{200,130},{220,150}})));

      Modelica.Blocks.Interfaces.RealOutput QHea_flow(final unit="W")
        "Electrical power consumed by the heating equipment" annotation (Placement(
            transformation(extent={{200,110},{220,130}}), iconTransformation(extent={{200,110},
                {220,130}})));

      Modelica.Blocks.Interfaces.RealOutput PCoo(final unit="W")
        "Electrical power consumed by the cooling equipment" annotation (Placement(
            transformation(extent={{200,90},{220,110}}), iconTransformation(extent={{200,90},
                {220,110}})));
      Modelica.Blocks.Interfaces.RealOutput PPum(final unit="W")
        "Electrical power consumed by the pumps"
        annotation (Placement(transformation(extent={{200,70},{220,90}}),
            iconTransformation(extent={{200,70},{220,90}})));

      Modelica.Blocks.Interfaces.RealOutput TMix(final unit="K", displayUnit="degC")
        "Mixed air temperature" annotation (Placement(transformation(extent={{200,-90},
                {220,-70}}), iconTransformation(extent={{200,-90},{220,-70}})));

      Modelica.Blocks.Interfaces.RealOutput TSup(
        final unit="K",
        displayUnit="degC") "Supply air temperature after coils"
        annotation (Placement(transformation(extent={{200,-130},{220,-110}}),
            iconTransformation(extent={{200,-130},{220,-110}})));

      Buildings.BoundaryConditions.WeatherData.Bus weaBus "Weather bus"
      annotation (Placement(
            transformation(extent={{-200,20},{-160,60}}),   iconTransformation(
              extent={{-170,128},{-150,148}})));

      Buildings.Fluid.Sensors.TemperatureTwoPort senTSup(
        m_flow_nominal=mAir_flow_nominal,
        allowFlowReversal=false,
        tau=0,
        redeclare package Medium = MediumA) "Supply air temperature sensor"
        annotation (Placement(transformation(extent={{128,30},{148,50}})));
      Buildings.Fluid.HeatExchangers.HeaterCooler_u heaCoi(
        m_flow_nominal=mAir_flow_nominal,
        Q_flow_nominal=QHea_flow_nominal,
        u(start=0),
        dp_nominal=0,
        allowFlowReversal=false,
        tau=90,
        redeclare package Medium = MediumA,
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
        show_T=true)
         "Air heating coil"
        annotation (Placement(transformation(extent={{52,30},{72,50}})));

      Buildings.Fluid.Movers.FlowControlled_m_flow fanSup(
        m_flow_nominal=mAir_flow_nominal,
        nominalValuesDefineDefaultPressureCurve=true,
        dp_nominal=875,
        per(use_powerCharacteristic=false),
        energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
        allowFlowReversal=false,
        use_inputFilter=false,
        redeclare package Medium = MediumA) "Supply fan"
        annotation (Placement(transformation(extent={{-30,30},{-10,50}})));

      Buildings.Fluid.FixedResistances.PressureDrop totalRes(
        m_flow_nominal=mAir_flow_nominal,
        dp_nominal=dp_nominal,
        allowFlowReversal=false,
        redeclare package Medium = MediumA)
        annotation (Placement(transformation(extent={{10,30},{30,50}})));

      Modelica.Blocks.Math.Gain eff(k=1/etaHea_nominal)
        annotation (Placement(transformation(extent={{120,110},{140,130}})));

      Buildings.Fluid.Sources.Outside out(
        nPorts=3,
        redeclare package Medium = MediumA)
        "Boundary conditions for outside air"
        annotation (Placement(transformation(extent={{-140,30},{-120,50}})));
      Buildings.Fluid.Sensors.TemperatureTwoPort senTMixAir(
        m_flow_nominal=mAir_flow_nominal,
        allowFlowReversal=false,
        tau=0,
        redeclare package Medium = MediumA) "Mixed air temperature sensor"
        annotation (Placement(transformation(extent={{-60,30},{-40,50}})));

      Buildings.Fluid.HeatExchangers.WetCoilCounterFlow      cooCoi(
        redeclare package Medium1 = MediumW,
        redeclare package Medium2 = MediumA,
        dp1_nominal=0,
        dp2_nominal=0,
        m2_flow_nominal=mAir_flow_nominal,
        allowFlowReversal1=false,
        allowFlowReversal2=false,
        m1_flow_nominal=mChiEva_flow_nominal,
        show_T=true,
        UA_nominal=-2*QCoo_flow_nominal/
            Buildings.Fluid.HeatExchangers.BaseClasses.lmtd(
                T_a1=27,
                T_b1=13,
                T_a2=6,
                T_b2=12))
        "Cooling coil"
        annotation (Placement(transformation(extent={{110,44},{90,24}})));

      Buildings.Fluid.Sources.MassFlowSource_T souChiWat(
        redeclare package Medium = MediumA,
        nPorts=1,
        use_T_in=true,
        m_flow=mChiCon_flow_nominal)
        "Mass flow source for chiller"
        annotation (Placement(transformation(
            extent={{10,-10},{-10,10}},
            origin={138,-174})));

      Buildings.Fluid.Movers.FlowControlled_m_flow pumChiWat(
        use_inputFilter=false,
        allowFlowReversal=false,
        redeclare package Medium = MediumW,
        energyDynamics=Modelica.Fluid.Types.Dynamics.SteadyState,
        m_flow_nominal=mChiEva_flow_nominal,
        addPowerToMedium=false,
        per(
          hydraulicEfficiency(eta={1}),
          motorEfficiency(eta={0.9}),
          motorCooledByFluid=false),
        dp_nominal=12000,
        inputType=Buildings.Fluid.Types.InputType.Continuous,
        nominalValuesDefineDefaultPressureCurve=true)
        "Pump for chilled water loop"
        annotation (
          Placement(transformation(
            extent={{-10,-10},{10,10}},
            rotation=90,
            origin={120,-90})));

      Buildings.Fluid.Chillers.ElectricEIR chi(
        allowFlowReversal1=false,
        allowFlowReversal2=false,
        redeclare package Medium1 = MediumA,
        redeclare package Medium2 = MediumW,
        m2_flow_nominal=mChiEva_flow_nominal,
        dp1_nominal=0,
        m1_flow_nominal=mChiCon_flow_nominal,
        per(
          capFunT={1.0433811,0.0407077,0.0004506,-0.0041514,-8.86e-5,-0.0003467},
          EIRFunT={9.946139E-01,-4.829399E-02,4.674277E-04,-1.158726E-03,5.762583E-04,2.148192E-04},
          EIRFunPLR={1.202277E-01,1.396384E-01,7.394038E-01},
          PLRMax=1.2,
          COP_nominal=COP_nominal,
          QEva_flow_nominal=QCoo_flow_nominal,
          mEva_flow_nominal=mChiEva_flow_nominal,
          mCon_flow_nominal=mChiCon_flow_nominal,
          TEvaLvg_nominal=TSupChi_nominal,
          PLRMinUnl=0.1,
          PLRMin=0.1,
          etaMotor=1,
          TEvaLvgMin=274.15,
          TEvaLvgMax=293.15,
          TConEnt_nominal=302.55,
          TConEntMin=274.15,
          TConEntMax=323.15),
        dp2_nominal=12E3,
        energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial)
        "Air cooled chiller"
        annotation (Placement(transformation(extent={{110,-158},{90,-178}})));

      Buildings.Fluid.Sources.Boundary_pT bouPreChi(
        redeclare package Medium = MediumW, nPorts=1)
        "Pressure boundary condition for chilled water loop"
        annotation (Placement(transformation(extent={{50,-172},{70,-152}})));

      Modelica.Blocks.Math.Gain gaiFan(k=mAir_flow_nominal)
        "Gain for fan mass flow rate"
        annotation (Placement(transformation(extent={{-80,130},{-60,150}})));

      IdealValve ideVal(
        redeclare package Medium = MediumW,
        final m_flow_nominal = mChiEva_flow_nominal) "Ideal valve"
        annotation (Placement(transformation(extent={{70,0},{90,20}})));

      Modelica.Blocks.Math.BooleanToReal booToInt(final realTrue=
            mChiEva_flow_nominal) "Boolean to integer conversion"
        annotation (Placement(transformation(extent={{60,-100},{80,-80}})));

      IdealValve ideEco(
        redeclare package Medium = MediumA,
        final m_flow_nominal=mAir_flow_nominal) "Ideal economizer" annotation (
          Placement(transformation(
            rotation=90,
            extent={{10,-10},{-10,10}},
            origin={-90,46})));
      Buildings.Fluid.Sensors.TemperatureTwoPort senTRetAir(
        m_flow_nominal=mAir_flow_nominal,
        allowFlowReversal=false,
        tau=0,
        redeclare package Medium = MediumA) "Return air temperature sensor"
        annotation (Placement(transformation(extent={{-20,-50},{-40,-30}})));
      Modelica.Blocks.Interfaces.RealOutput TRet(final unit="K", displayUnit="degC")
        "Return air temperature" annotation (Placement(transformation(extent={{200,
                -110},{220,-90}}), iconTransformation(extent={{200,-110},{220,-90}})));
    equation
      connect(fanSup.port_b, totalRes.port_a)
        annotation (Line(points={{-10,40},{10,40}},  color={0,127,255}));
      connect(fanSup.P, PFan) annotation (Line(points={{-9,49},{-6,49},{-6,140},{210,
              140}},             color={0,0,127}));
      connect(eff.y, QHea_flow) annotation (Line(points={{141,120},{210,120}},
                          color={0,0,127}));
      connect(weaBus, out.weaBus) annotation (Line(
          points={{-180,40},{-140,40},{-140,40.2}},
          color={255,204,51},
          thickness=0.5), Text(
          textString="%first",
          index=-1,
          extent={{-6,3},{-6,3}}));
      connect(senTMixAir.port_b, fanSup.port_a)
        annotation (Line(points={{-40,40},{-30,40}},          color={0,127,255}));
      connect(heaCoi.Q_flow, eff.u) annotation (Line(points={{73,46},{80,46},{80,
              120},{118,120}},                        color={0,0,127}));
      connect(heaCoi.port_b, cooCoi.port_a2)
        annotation (Line(points={{72,40},{90,40}}, color={0,127,255}));
      connect(cooCoi.port_b2, senTSup.port_a)
        annotation (Line(points={{110,40},{128,40}},          color={0,127,255}));
      connect(cooCoi.port_b1, ideVal.port_1) annotation (Line(
          points={{90,28},{86,28},{86,19.8}},
          color={0,0,255},
          thickness=0.5));
      connect(chi.port_b2, pumChiWat.port_a) annotation (Line(points={{110,-162},{120,
              -162},{120,-100}},            color={0,0,255},
          thickness=0.5));
      connect(souChiWat.ports[1], chi.port_a1) annotation (Line(points={{128,-174},
              {128,-174},{110,-174}},       color={0,127,255}));
      connect(chi.port_b1, out.ports[1]) annotation (Line(points={{90,-174},{
              -116,-174},{-116,42.6667},{-120,42.6667}},             color={0,127,255}));
      connect(weaBus.TDryBul, souChiWat.T_in) annotation (Line(
          points={{-180,40},{-180,-208},{160,-208},{160,-170},{150,-170}},
          color={255,204,51},
          thickness=0.5), Text(
          textString="%first",
          index=-1,
          extent={{-6,3},{-6,3}}));

      connect(pumChiWat.P, PPum) annotation (Line(points={{111,-79},{111,-52},{180,-52},
              {180,80},{210,80}},      color={0,0,127}));
      connect(chi.P, PCoo) annotation (Line(points={{89,-177},{84,-177},{84,-128},{98,
              -128},{98,-50},{178,-50},{178,100},{210,100}},
            color={0,0,127}));
      connect(ideVal.port_2, chi.port_a2)
        annotation (Line(points={{86,0.2},{86,-162},{90,-162}},
                                                              color={0,127,255}));
      connect(cooCoi.port_a1, pumChiWat.port_b) annotation (Line(points={{110,28},{120,
              28},{120,-80}},              color={0,127,255}));
      connect(cooCoi.port_a1, ideVal.port_3) annotation (Line(points={{110,28},{120,
              28},{120,10},{90,10}}, color={0,127,255}));
      connect(bouPreChi.ports[1], chi.port_a2) annotation (Line(points={{70,-162},{90,
              -162}},              color={0,127,255}));
      connect(totalRes.port_b, heaCoi.port_a)
        annotation (Line(points={{30,40},{52,40}}, color={0,127,255}));
      connect(senTSup.port_b, supplyAir) annotation (Line(points={{148,40},{200,40}},
                                  color={0,127,255}));
      connect(gaiFan.y, fanSup.m_flow_in)
        annotation (Line(points={{-59,140},{-20,140},{-20,52}}, color={0,0,127}));

    protected
      model IdealValve
        extends Modelica.Blocks.Icons.Block;

        replaceable package Medium = Modelica.Media.Interfaces.PartialMedium "Medium in the component"
            annotation (choicesAllMatching = true);

        parameter Modelica.SIunits.MassFlowRate m_flow_nominal
          "Design chilled water supply flow";
        Modelica.Fluid.Interfaces.FluidPort_a port_1(redeclare package Medium =
              Medium) annotation (Placement(transformation(extent={{50,88},
                  {70,108}}), iconTransformation(extent={{50,88},{70,108}})));
        Modelica.Fluid.Interfaces.FluidPort_b port_2(redeclare package Medium =
              Medium) annotation (Placement(transformation(extent={{50,-108},
                  {70,-88}}), iconTransformation(extent={{50,-108},{70,-88}})));
        Modelica.Fluid.Interfaces.FluidPort_a port_3(redeclare package Medium =
              Medium) annotation (Placement(transformation(extent={{90,-10},
                  {110,10}}), iconTransformation(extent={{90,-10},{110,10}})));
        Modelica.Blocks.Interfaces.RealInput y(min=0, max=1) annotation (Placement(
              transformation(extent={{-120,-10},{-100,10}}),
              iconTransformation(extent={{-120,-10},{-100,10}})));
        Buildings.Fluid.Sensors.MassFlowRate senMasFlo(redeclare package Medium =
              Medium, allowFlowReversal=false) "Mass flow rate sensor"
          annotation (Placement(transformation(
              extent={{10,-10},{-10,10}},
              rotation=90,
              origin={0,-40})));
        Buildings.Fluid.Movers.BaseClasses.IdealSource preMasFlo(
          redeclare package Medium = Medium,
          control_m_flow=true,
          control_dp=false,
          m_flow_small=m_flow_nominal*1E-5,
          show_V_flow=false,
          allowFlowReversal=false) "Prescribed mass flow rate for the bypass"
          annotation (Placement(transformation(
              extent={{-10,10},{10,-10}},
              rotation=180,
              origin={50,0})));
        Modelica.Blocks.Math.Product pro "Product for mass flow rate computation"
          annotation (Placement(transformation(extent={{-28,6},{-8,26}})));
        Modelica.Blocks.Sources.Constant one(final k=1) "Outputs one"
          annotation (Placement(transformation(extent={{-90,12},{-70,32}})));
        Modelica.Blocks.Math.Feedback feedback
          annotation (Placement(transformation(extent={{-60,12},{-40,32}})));
      equation
        connect(senMasFlo.m_flow, pro.u2) annotation (Line(points={{-11,-40},{-40,
                -40},{-40,10},{-30,10}},      color={0,0,127}));
        connect(feedback.u1, one.y)
          annotation (Line(points={{-58,22},{-69,22}},
                                                     color={0,0,127}));
        connect(y, feedback.u2)
          annotation (Line(points={{-110,0},{-50,0},{-50,14}},color={0,0,127}));
        connect(preMasFlo.port_a, port_3)
          annotation (Line(points={{60,-1.33227e-15},{80,-1.33227e-15},{80,0},{100,
                0}},                                   color={0,127,255}));
        connect(feedback.y, pro.u1)
          annotation (Line(points={{-41,22},{-30,22}},
                                                     color={0,0,127}));
        connect(pro.y, preMasFlo.m_flow_in)
          annotation (Line(points={{-7,16},{56,16},{56,8}},    color={0,0,127}));
        connect(port_1, senMasFlo.port_a)
          annotation (Line(points={{60,98},{60,60},{4.44089e-16,60},{4.44089e-16,
                -30}},                                  color={0,127,255}));
        connect(senMasFlo.port_b, port_2)
          annotation (Line(points={{-4.44089e-16,-50},{0,-50},{0,-72},{60,-72},{60,
                -92},{60,-92},{60,-98},{60,-98}},      color={0,127,255}));
        connect(preMasFlo.port_b, senMasFlo.port_a) annotation (Line(points={{40,
                1.33227e-15},{4.44089e-16,1.33227e-15},{4.44089e-16,-30}},
                                        color={0,127,255}));
        annotation (
          Icon(
            graphics={
              Polygon(
                points={{60,0},{68,14},{52,14},{60,0}},
                lineColor={0,0,0},
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid),
              Line(points={{60,100},{60,-100}}, color={28,108,200}),
              Line(points={{102,0},{62,0}}, color={28,108,200}),
              Polygon(
                points={{60,0},{68,-14},{52,-14},{60,0}},
                lineColor={0,0,0},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid),
              Line(points={{62,0},{-98,0}}, color={0,0,0}),
              Rectangle(
                visible=use_inputFilter,
                extent={{28,-10},{46,10}},
                lineColor={0,0,0},
                fillColor={135,135,135},
                fillPattern=FillPattern.Solid),
              Polygon(
                points={{72,-8},{72,8},{60,0},{72,-8}},
                lineColor={0,0,0},
                fillColor={0,0,0},
                fillPattern=FillPattern.Solid)}));
      end IdealValve;
    equation
      connect(booToInt.y, pumChiWat.m_flow_in)
        annotation (Line(points={{81,-90},{108,-90}}, color={0,0,127}));
      connect(booToInt.u, chiOn) annotation (Line(points={{58,-90},{40,-90},{40,-140},
              {-220,-140}}, color={255,0,255}));
      connect(chiOn, chi.on) annotation (Line(points={{-220,-140},{40,-140},{40,-188},
              {120,-188},{120,-171},{112,-171}}, color={255,0,255}));
      connect(gaiFan.u, uFan)
        annotation (Line(points={{-82,140},{-220,140}}, color={0,0,127}));
      connect(heaCoi.u, uHea) annotation (Line(points={{50,46},{40,46},{40,80},{-220,
              80}}, color={0,0,127}));
      connect(ideVal.y, uCooVal)
        annotation (Line(points={{69,10},{-220,10}}, color={0,0,127}));
      connect(chi.TSet, TSetChi) annotation (Line(points={{112,-165},{124,-165},{124,
              -190},{-220,-190}}, color={0,0,127}));
      connect(senTMixAir.T, TMix) annotation (Line(points={{-50,51},{-50,70},{188,
              70},{188,-80},{210,-80}}, color={0,0,127}));
      connect(senTSup.T, TSup) annotation (Line(points={{138,51},{138,64},{170,64},
              {170,-120},{210,-120}},color={0,0,127}));
      connect(out.ports[2], ideEco.port_1) annotation (Line(points={{-120,40},{-120,
              40},{-99.8,40}},             color={0,127,255}));
      connect(ideEco.port_2, senTMixAir.port_a)
        annotation (Line(points={{-80.2,40},{-60,40}}, color={0,127,255}));
      connect(uEco, ideEco.y) annotation (Line(points={{-220,-60},{-148,-60},{-148,70},
              {-90,70},{-90,57}}, color={0,0,127}));
      connect(senTRetAir.port_a, returnAir)
        annotation (Line(points={{-20,-40},{200,-40}}, color={0,127,255}));
      connect(ideEco.port_3, senTRetAir.port_b) annotation (Line(points={{-90,36},{
              -90,-40},{-40,-40}}, color={0,127,255}));
      connect(senTRetAir.port_b, out.ports[3]) annotation (Line(points={{-40,-40},
              {-112,-40},{-112,36},{-120,36},{-120,37.3333}},color={0,127,255}));
      connect(TRet, senTRetAir.T) annotation (Line(points={{210,-100},{174,-100},{
              174,-20},{-30,-20},{-30,-29}}, color={0,0,127}));
      annotation (Icon(coordinateSystem(preserveAspectRatio=false, extent={{-200,-240},
                {200,160}}), graphics={
            Rectangle(
              extent={{-202,160},{200,-240}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{180,40},{-160,0}},
              lineColor={175,175,175},
              fillColor={175,175,175},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{-32,36},{-4,22}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{180,-72},{-160,-112}},
              lineColor={175,175,175},
              fillColor={175,175,175},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{-80,0},{-120,-72}},
              lineColor={175,175,175},
              fillColor={175,175,175},
              fillPattern=FillPattern.Solid),
            Ellipse(
              extent={{-48,36},{-14,2}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Ellipse(
              extent={{-38,26},{-24,12}},
              lineColor={0,0,0},
              fillColor={0,0,0},
              fillPattern=FillPattern.Solid),
            Rectangle(
              extent={{40,40},{54,0}},
              lineColor={255,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Backward),
            Rectangle(
              extent={{102,40},{116,0}},
              lineColor={0,0,255},
              fillColor={255,255,255},
              fillPattern=FillPattern.Backward),
            Rectangle(
              extent={{42,54},{52,46}},
              lineColor={0,0,0},
              fillColor={0,0,0},
              fillPattern=FillPattern.Backward),
            Rectangle(
              extent={{38,56},{56,54}},
              lineColor={0,0,0},
              fillColor={0,0,0},
              fillPattern=FillPattern.Backward),
            Line(points={{44,56},{44,60}}, color={0,0,0}),
            Line(points={{50,56},{50,60}}, color={0,0,0}),
            Line(points={{48,40},{48,48}}, color={0,0,0}),
            Rectangle(
              extent={{-140,40},{-126,0}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Backward),
            Rectangle(
              extent={{-140,-72},{-126,-112}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Backward),
            Rectangle(
              extent={{-7,20},{7,-20}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Backward,
              origin={-100,-37},
              rotation=90),
            Line(points={{200,100},{86,100},{86,46}},   color={0,0,127}),
            Line(points={{198,118},{48,118},{48,68}}, color={0,0,127}),
            Line(points={{198,140},{-30,140},{-30,50}}, color={0,0,127}),
            Line(points={{104,0},{104,-66}}, color={0,0,255}),
            Line(points={{114,0},{114,-66}}, color={0,0,255}),
            Line(points={{104,-26},{114,-26}}, color={0,0,255}),
            Polygon(
              points={{-3,4},{-3,-4},{3,0},{-3,4}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid,
              origin={115,-24},
              rotation=-90),
            Polygon(
              points={{110,-22},{110,-30},{116,-26},{110,-22}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Polygon(
              points={{-4,-3},{4,-3},{0,3},{-4,-3}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid,
              origin={115,-28}),
            Line(points={{116,-26},{122,-26}}, color={0,0,0}),
            Line(points={{122,-24},{122,-30}}, color={0,0,0}),
            Ellipse(
              extent={{96,-124},{124,-152}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Polygon(
              points={{110,-124},{98,-144},{122,-144},{110,-124}},
              lineColor={0,0,0},
              fillColor={0,0,0},
              fillPattern=FillPattern.Solid),
            Line(points={{114,-116},{114,-124}},
                                             color={0,0,255}),
            Line(points={{104,-116},{104,-124}},
                                             color={0,0,255}),
            Ellipse(
              extent={{84,-148},{110,-158}},
              lineColor={0,0,0},
              fillColor={95,95,95},
              fillPattern=FillPattern.Solid),
            Ellipse(
              extent={{110,-148},{136,-158}},
              lineColor={0,0,0},
              fillColor={95,95,95},
              fillPattern=FillPattern.Solid),
            Ellipse(
              extent={{108,-48},{120,-58}},
              lineColor={0,0,0},
              fillColor={255,255,255},
              fillPattern=FillPattern.Solid),
            Polygon(
              points={{114,-48},{110,-56},{118,-56},{114,-48}},
              lineColor={0,0,0},
              fillColor={0,0,0},
              fillPattern=FillPattern.Solid),
            Line(points={{200,80},{132,80},{132,46}},   color={0,0,127}),
            Line(points={{124,-54},{132,-54},{132,-4}}, color={0,0,127}),
            Line(points={{92,-136},{86,-136},{86,-4}},  color={0,0,127})}),
                                                                     Diagram(
            coordinateSystem(preserveAspectRatio=false, extent={{-200,-240},{200,160}})),
          Documentation(info="<html>
<p>
This is a conventional single zone VAV HVAC system model. The system contains
a variable speed supply fan, electric heating coil, water-based cooling coil,
economizer, and air-cooled chiller. The control of the system is that of
conventional VAV heating and cooling. During cooling, the supply air
temperature is held constant while the supply air flow is modulated from
maximum to minimum according to zone load. This is done by modulating the
fan speed. During heating, the supply air flow is held at a constant minimum
while the heating coil is modulated accoding to zone load. The mass flow of
chilled water through the cooling coil is controlled by a three-way valve to
maintain the supply air temperature setpoint during cooling.
The mixing box maintains the minimum outside airflow fraction unless
conditions for economizer are met, in which case the economizer controller
adjusts the outside airflow fraction to meet a mixed air temperature setpoint.
The economizer is enabled if the outside air drybulb temperature is lower
than the return air temperature and the system is not in heating mode.
</p>
<p>
There are a number of assumptions in the model. Pressure drops through the
system are collected into a single component. The mass flow of return air
is equal to the mass flow of supply air. The mass flow of outside air and
relief air in the mixing box is ideally controlled so that the supply air is
composed of the specified outside airflow fraction, rather than having
feedback control of damper positions. The cooling coil is a dry coil model.
</p>
</html>",     revisions="<html>
<ul>
<li>
September 08, 2017, by Thierry S. Nouidui:<br/>
Removed experiment annotation.
</li>
<li>
June 21, 2017, by Michael Wetter:<br/>
Refactored implementation.
</li>
<li>
June 1, 2017, by David Blum:<br/>
First implementation.
</li>
</ul>
</html>"));
    end ZoneTemperatureBase;

    package Control "VAV system model"
      extends Modelica.Icons.Package;

      model ChillerDXHeatingEconomizerController
        "Controller for single zone VAV system"
        extends Modelica.Blocks.Icons.Block;

        parameter Modelica.SIunits.Temperature TSupChi_nominal
          "Design value for chiller leaving water temperature";
        parameter Real minAirFlo(
          min=0,
          max=1,
          unit="1") = 0.2
          "Minimum airflow rate of system"
          annotation(Dialog(group="Air design"));

        parameter Modelica.SIunits.DimensionlessRatio minOAFra "Minimum outdoor air fraction of system"
          annotation(Dialog(group="Air design"));

        parameter Modelica.SIunits.Temperature TSetSupAir "Cooling supply air temperature setpoint"
          annotation(Dialog(group="Air design"));

        parameter Real kHea(min=Modelica.Constants.small) = 2
          "Gain of heating controller"
          annotation(Dialog(group="Control gain"));

        parameter Real kCoo(min=Modelica.Constants.small)=1
          "Gain of controller for cooling valve"
          annotation(Dialog(group="Control gain"));

        parameter Real kFan(min=Modelica.Constants.small) = 0.5
          "Gain of controller for fan"
          annotation(Dialog(group="Control gain"));

        parameter Real kEco(min=Modelica.Constants.small) = 4
          "Gain of controller for economizer"
          annotation(Dialog(group="Control gain"));

        Modelica.Blocks.Interfaces.RealInput TRoo(
          final unit="K",
          displayUnit="degC") "Zone temperature measurement"
        annotation (Placement(
              transformation(
              extent={{-20,-20},{20,20}},
              origin={-120,-60})));

        Modelica.Blocks.Interfaces.RealInput TSetRooCoo(
          final unit="K",
          displayUnit="degC")
          "Zone cooling setpoint temperature" annotation (Placement(transformation(
              extent={{20,-20},{-20,20}},
              rotation=180,
              origin={-120,30})));
        Modelica.Blocks.Interfaces.RealInput TSetRooHea(
          final unit="K",
          displayUnit="degC")
          "Zone heating setpoint temperature" annotation (Placement(transformation(
              extent={{20,-20},{-20,20}},
              rotation=180,
              origin={-120,60})));

        Modelica.Blocks.Interfaces.RealInput TMix(
          final unit="K",
          displayUnit="degC")
          "Measured mixed air temperature"
          annotation (Placement(transformation(extent={{-140,-20},{-100,20}})));

        Modelica.Blocks.Interfaces.RealInput TSup(
          final unit="K",
          displayUnit="degC")
          "Measured supply air temperature after the cooling coil"
          annotation (Placement(transformation(extent={{-140,-110},{-100,-70}})));

        Modelica.Blocks.Interfaces.RealInput TOut(
          final unit="K",
          displayUnit="degC")
          "Measured outside air temperature"
          annotation (Placement(transformation(extent={{-140,-50},{-100,-10}})));

        Modelica.Blocks.Interfaces.RealOutput yHea(final unit="1") "Control signal for heating coil"
          annotation (Placement(transformation(extent={{100,50},{120,70}})));

        Modelica.Blocks.Interfaces.RealOutput yFan(final unit="1") "Control signal for fan"
          annotation (Placement(transformation(extent={{100,80},{120,100}})));

        Modelica.Blocks.Interfaces.RealOutput yOutAirFra(final unit="1")
          "Control signal for outside air fraction"
          annotation (Placement(transformation(extent={{100,20},{120,40}})));

        Modelica.Blocks.Interfaces.RealOutput yCooCoiVal(final unit="1")
          "Control signal for cooling coil valve"
          annotation (Placement(transformation(extent={{100,-10},{120,10}})));

        Modelica.Blocks.Interfaces.RealOutput TSetSupChi(
          final unit="K",
          displayUnit="degC")
          "Set point for chiller leaving water temperature"
          annotation (Placement(transformation(extent={{100,-90},{120,-70}})));

        Modelica.Blocks.Interfaces.BooleanOutput chiOn "On signal for chiller"
          annotation (Placement(transformation(extent={{100,-50},{120,-30}})));

        BaseClasses.ControllerHeatingFan conSup(
          minAirFlo = minAirFlo,
          kHea = kHea,
          kFan = kFan) "Heating coil, cooling coil and fan controller"
          annotation (Placement(transformation(extent={{-40,70},{-20,90}})));
        BaseClasses.ControllerEconomizer conEco(
          final kEco = kEco)
          "Economizer control"
          annotation (Placement(transformation(extent={{0,40},{20,60}})));

        Buildings.Controls.Continuous.LimPID conCooVal(
          controllerType=Modelica.Blocks.Types.SimpleController.PI,
          Ti=240,
          final yMax=1,
          final k=kCoo,
          final reverseAction=true,
          reset=Buildings.Types.Reset.Parameter)
                                    "Cooling coil valve controller"
          annotation (Placement(transformation(extent={{0,-40},{20,-20}})));

        BaseClasses.FanStatus fanSta
          annotation (Placement(transformation(extent={{0,10},{20,30}})));
        Buildings.Controls.OBC.CDL.Interfaces.BooleanInput uOcc
          "Current occupancy period, true if it is in occupant period"
          annotation (Placement(transformation(extent={{-140,70},{-100,110}}),
              iconTransformation(extent={{-180,50},{-100,130}})));

        Modelica.Blocks.Logical.Switch TSupAirSetSwi
          annotation (Placement(transformation(extent={{-40,-30},{-20,-10}})));
        Modelica.Blocks.Logical.Switch fanSpeSwi
          annotation (Placement(transformation(extent={{68,80},{88,100}})));
        Modelica.Blocks.Logical.Switch yHeaSwi
          annotation (Placement(transformation(extent={{68,50},{88,70}})));
      protected
        Modelica.Blocks.Sources.Constant TSetSupChiConst(
          final k=TSupChi_nominal)
          "Set point for chiller temperature"
          annotation (Placement(transformation(extent={{40,-90},{60,-70}})));

        Modelica.Blocks.Sources.Constant conMinOAFra(
          final k=minOAFra) "Minimum outside air fraction"
          annotation (Placement(transformation(extent={{-70,38},{-50,58}})));

        Modelica.Blocks.Sources.Constant TSetSupAirOff(final k=TSetSupAir + 10)
          "Set point for supply air temperature"
          annotation (Placement(transformation(extent={{40,-20},{60,0}})));

        Modelica.Blocks.Sources.Constant TSetSupAirOn(final k=TSetSupAir)
          "Set point for supply air temperature"
          annotation (Placement(transformation(extent={{40,16},{60,36}})));
        Modelica.Blocks.Sources.Constant zer(final k=0) "0"
          annotation (Placement(transformation(extent={{32,66},{52,86}})));
      equation
        connect(conMinOAFra.y,conEco. minOAFra) annotation (Line(points={{-49,48},{
                -26,48},{-1,48}},                 color={0,0,127}));
        connect(conSup.TSetRooHea, TSetRooHea) annotation (Line(points={{-42,83},
                {-88,83},{-88,60},{-120,60}},
                                         color={0,0,127}));
        connect(conSup.TSetRooCoo, TSetRooCoo) annotation (Line(points={{-42,78},
                {-80,78},{-80,30},{-120,30}},
                                         color={0,0,127}));
        connect(conSup.TRoo, TRoo) annotation (Line(points={{-42,73},{-74,73},{
                -74,-60},{-120,-60}},
                             color={0,0,127}));
        connect(conSup.yHea, conEco.yHea) annotation (Line(points={{-19,76},{-10,76},
                {-10,42},{-1,42}},color={0,0,127}));
        connect(conEco.TMix, TMix) annotation (Line(points={{-1,55},{-40,55},{
                -40,0},{-120,0}},
                            color={0,0,127}));
        connect(conEco.TRet, TRoo) annotation (Line(points={{-1,52},{-34,52},{-34,12},
                {-88,12},{-88,-60},{-120,-60}},     color={0,0,127}));
        connect(conEco.TOut, TOut) annotation (Line(points={{-1,45},{-30,45},{
                -30,8},{-94,8},{-94,-30},{-120,-30}},
                                                 color={0,0,127}));
        connect(conEco.yOutAirFra, yOutAirFra) annotation (Line(points={{21,50},{80,50},
                {80,30},{110,30}}, color={0,0,127}));
        connect(conCooVal.y, yCooCoiVal)
          annotation (Line(points={{21,-30},{76,-30},{76,0},{110,0}},
                                                    color={0,0,127}));
        connect(TSetSupChiConst.y, TSetSupChi)
          annotation (Line(points={{61,-80},{110,-80}}, color={0,0,127}));
        connect(conCooVal.u_m, TSup)
          annotation (Line(points={{10,-42},{10,-90},{-120,-90}}, color={0,0,127}));
        connect(fanSta.uOcc, uOcc) annotation (Line(points={{-4,28},{-12,28},{-12,100},
                {-80,100},{-80,90},{-120,90}},          color={255,0,255}));
        connect(TSetRooHea, fanSta.TSetRooHea) annotation (Line(points={{-120,
                60},{-88,60},{-88,23},{-2,23}}, color={0,0,127}));
        connect(TSetRooCoo, fanSta.TSetRooCoo) annotation (Line(points={{-120,
                30},{-80,30},{-80,17},{-2,17}}, color={0,0,127}));
        connect(TRoo, fanSta.TRoo) annotation (Line(points={{-120,-60},{-68,-60},
                {-68,12},{-2,12}}, color={0,0,127}));
        connect(fanSta.fanOn, TSupAirSetSwi.u2) annotation (Line(points={{21,20},
                {28,20},{28,0},{-54,0},{-54,-20},{-42,-20}}, color={255,0,255}));
        connect(TSetSupAirOff.y, TSupAirSetSwi.u3) annotation (Line(points={{61,
                -10},{72,-10},{72,6},{-58,6},{-58,-28},{-42,-28}}, color={0,0,
                127}));
        connect(TSetSupAirOn.y, TSupAirSetSwi.u1) annotation (Line(points={{61,26},{72,
                26},{72,8},{-50,8},{-50,-12},{-42,-12}},         color={0,0,127}));
        connect(TSupAirSetSwi.y, conEco.TMixSet) annotation (Line(points={{-19,
                -20},{-14,-20},{-14,58},{-1,58}}, color={0,0,127}));
        connect(TSupAirSetSwi.y, conCooVal.u_s) annotation (Line(points={{-19,
                -20},{-14,-20},{-14,-30},{-2,-30}}, color={0,0,127}));
        connect(fanSta.fanOn, fanSpeSwi.u2) annotation (Line(points={{21,20},{
                28,20},{28,90},{66,90}}, color={255,0,255}));
        connect(conSup.yFan, fanSpeSwi.u1) annotation (Line(points={{-19,84},{
                20,84},{20,98},{66,98}}, color={0,0,127}));
        connect(zer.y, fanSpeSwi.u3) annotation (Line(points={{53,76},{60,76},{
                60,82},{66,82}}, color={0,0,127}));
        connect(fanSpeSwi.y, yFan)
          annotation (Line(points={{89,90},{110,90}}, color={0,0,127}));
        connect(conSup.yHea, yHeaSwi.u1) annotation (Line(points={{-19,76},{20,76},{20,
                68},{66,68}}, color={0,0,127}));
        connect(fanSta.fanOn, yHeaSwi.u2) annotation (Line(points={{21,20},{28,20},{28,
                60},{66,60}}, color={255,0,255}));
        connect(zer.y, yHeaSwi.u3) annotation (Line(points={{53,76},{60,76},{60,52},{66,
                52}}, color={0,0,127}));
        connect(yHeaSwi.y, yHea)
          annotation (Line(points={{89,60},{110,60}}, color={0,0,127}));
        connect(fanSta.fanOn, conSup.trigger) annotation (Line(points={{21,20},
                {28,20},{28,94},{-50,94},{-50,88},{-42,88}}, color={255,0,255}));
        connect(fanSta.fanOn, conCooVal.trigger) annotation (Line(points={{21,
                20},{28,20},{28,-4},{-54,-4},{-54,-46},{2,-46},{2,-42}}, color=
                {255,0,255}));
        connect(fanSta.fanOn, chiOn) annotation (Line(points={{21,20},{28,20},{
                28,-40},{110,-40}}, color={255,0,255}));
        annotation (Icon(graphics={Line(points={{-100,-100},{0,2},{-100,100}}, color=
                    {0,0,0})}), Documentation(info="<html>
<p>
This is the controller for the VAV system with economizer, heating coil and cooling coil.
</p>
</html>",       revisions="<html>
<ul>
<li>
June 21, 2017, by Michael Wetter:<br/>
Refactored implementation.
</li>
<li>
June 1, 2017, by David Blum:<br/>
First implementation.
</li>
</ul>
</html>"));
      end ChillerDXHeatingEconomizerController;

      package BaseClasses "Package with base classes for Buildings.Air.Systems.SingleZone.VAV"
        extends Modelica.Icons.BasesPackage;

        model ControllerEconomizer "Controller for economizer"
          extends Modelica.Blocks.Icons.Block;

          parameter Real kEco(min=Modelica.Constants.small) = 1
            "Gain of controller"
            annotation(Dialog(group="Control gain"));

          Modelica.Blocks.Interfaces.RealInput TMixSet(
            final unit="K",
            displayUnit="degC")
            "Mixed air setpoint temperature"
            annotation (Placement(transformation(extent={{-120,70},{-100,90}})));
          Modelica.Blocks.Interfaces.RealInput TMix(
            final unit="K",
            displayUnit="degC")
            "Measured mixed air temperature"
            annotation (Placement(transformation(extent={{-120,40},{-100,60}})));

          Modelica.Blocks.Interfaces.RealInput TOut(
            final unit="K",
            displayUnit="degC")
            "Measured outside air temperature"
            annotation (Placement(
                transformation(extent={{-120,-60},{-100,-40}})));
          Modelica.Blocks.Interfaces.RealInput yHea(final unit="1")
            "Control signal for heating coil" annotation (Placement(transformation(
                  extent={{-120,-90},{-100,-70}})));

          Modelica.Blocks.Interfaces.RealInput TRet(
            final unit="K",
            displayUnit="degC")
            "Return air temperature"
            annotation (Placement(transformation(extent={{-120,10},{-100,30}})));

          Modelica.Blocks.Interfaces.RealInput minOAFra(
            min = 0,
            max = 1,
            final unit="1")
            "Minimum outside air fraction"
            annotation (Placement(transformation(extent={{-120,-30},{-100,-10}})));

          Modelica.Blocks.Interfaces.RealOutput yOutAirFra(final unit="1")
            "Control signal for outside air fraction"
            annotation (Placement(transformation(extent={{100,-10},{120,10}})));

          Modelica.Blocks.Nonlinear.VariableLimiter Limiter(strict=true)
            "Signal limiter"
            annotation (Placement(transformation(extent={{60,-10},{80,10}})));
          Modelica.Blocks.Sources.Constant const(final k=1)
            "Constant output signal with value 1"
            annotation (Placement(transformation(extent={{20,60},{40,80}})));

          Modelica.Blocks.Logical.Switch switch1 "Switch to select control output"
            annotation (Placement(transformation(extent={{20,10},{40,30}})));

          Modelica.Blocks.MathBoolean.And and1(final nu=3) "Logical and"
            annotation (Placement(transformation(extent={{20,-60},{40,-40}})));
          Buildings.Controls.Continuous.LimPID con(
            final k=kEco,
            final reverseAction=true,
            final yMax=Modelica.Constants.inf,
            final yMin=-Modelica.Constants.inf,
            controllerType=Modelica.Blocks.Types.SimpleController.P)
            "Controller"
            annotation (Placement(transformation(extent={{-90,70},{-70,90}})));
          Modelica.Blocks.Math.Feedback feedback "Control error"
            annotation (Placement(transformation(extent={{-50,-38},{-30,-18}})));
          Buildings.Controls.OBC.CDL.Continuous.HysteresisWithHold hysYHea(
            trueHoldDuration=60*15,
            uLow=0.05,
            uHigh=0.15) "Hysteresis with delay for heating signal"
            annotation (Placement(transformation(extent={{-80,-90},{-60,-70}})));
          Buildings.Controls.OBC.CDL.Continuous.HysteresisWithHold hysTMix(
            uLow=-0.5,
            uHigh=0.5,
            trueHoldDuration=60*15)
            "Hysteresis with delay for mixed air temperature"
            annotation (Placement(transformation(extent={{-20,-60},{0,-40}})));
          Modelica.Blocks.Logical.Not not1
            annotation (Placement(transformation(extent={{-40,-90},{-20,-70}})));

          Modelica.Blocks.Math.Feedback feedback1
            annotation (Placement(transformation(extent={{-70,20},{-50,40}})));
          Buildings.Controls.OBC.CDL.Continuous.HysteresisWithHold hysCooPot(
            uHigh=0.5,
            uLow=0,
            trueHoldDuration=60*15)
            "Hysteresis with delay to check for cooling potential of outside air"
            annotation (Placement(transformation(extent={{-40,20},{-20,40}})));
        equation
          connect(Limiter.limit2, minOAFra) annotation (Line(points={{58,-8},{-20,-8},{
                  -20,-8},{-94,-8},{-94,-20},{-110,-20},{-110,-20}},
                                        color={0,0,127}));
          connect(const.y, Limiter.limit1) annotation (Line(points={{41,70},{50,70},{50,
                  8},{58,8}},           color={0,0,127}));
          connect(minOAFra, switch1.u3) annotation (Line(points={{-110,-20},{-94,-20},{
                  -94,12},{18,12}},  color={0,0,127}));
          connect(switch1.y, Limiter.u) annotation (Line(points={{41,20},{46,20},{46,0},
                  {58,0}},          color={0,0,127}));
          connect(and1.y, switch1.u2) annotation (Line(points={{41.5,-50},{48,-50},{48,
                  -6},{10,-6},{10,20},{18,20}},
                                             color={255,0,255}));
          connect(con.u_s, TMixSet)
            annotation (Line(points={{-92,80},{-92,80},{-110,80}}, color={0,0,127}));
          connect(TMix, con.u_m)
            annotation (Line(points={{-110,50},{-80,50},{-80,68}}, color={0,0,127}));
          connect(con.y, switch1.u1) annotation (Line(points={{-69,80},{12,80},{12,28},
                  {18,28}}, color={0,0,127}));
          connect(TOut, feedback.u2) annotation (Line(points={{-110,-50},{-40,-50},{-40,
                  -36}}, color={0,0,127}));
          connect(feedback.u1, TMix) annotation (Line(points={{-48,-28},{-80,-28},{-80,
                  50},{-110,50}}, color={0,0,127}));
          connect(Limiter.y, yOutAirFra)
            annotation (Line(points={{81,0},{110,0}}, color={0,0,127}));
          connect(hysYHea.u, yHea)
            annotation (Line(points={{-82,-80},{-110,-80}}, color={0,0,127}));
          connect(feedback.y, hysTMix.u)
            annotation (Line(points={{-31,-28},{-28,-28},{-28,-50},{-22,-50}},
                                                           color={0,0,127}));
          connect(feedback1.u1, TRet)
            annotation (Line(points={{-68,30},{-88,30},{-88,20},{-110,20}},
                                                          color={0,0,127}));
          connect(feedback1.u2,TOut)
            annotation (Line(points={{-60,22},{-60,-50},{-110,-50}}, color={0,0,127}));
          connect(feedback1.y, hysCooPot.u)
            annotation (Line(points={{-51,30},{-42,30}}, color={0,0,127}));
          connect(hysCooPot.y, and1.u[1]) annotation (Line(points={{-18,30},{6,
                  30},{6,-45.3333},{20,-45.3333}},
                                            color={255,0,255}));
          connect(hysTMix.y, and1.u[2])
            annotation (Line(points={{2,-50},{20,-50},{20,-50}}, color={255,0,255}));
          connect(not1.y, and1.u[3]) annotation (Line(points={{-19,-80},{-19,
                  -80},{6,-80},{6,-54.6667},{20,-54.6667}},
                                                    color={255,0,255}));
          connect(hysYHea.y, not1.u) annotation (Line(points={{-58,-80},{-42,
                  -80},{-42,-80}},
                         color={255,0,255}));
          annotation (    Documentation(info="<html>
<p>
Economizer controller.
</p>
</html>",         revisions="<html>
<ul>
<li>
June 21, 2017, by Michael Wetter:<br/>
Refactored implementation.
</li>
<li>
June 1, 2017, by David Blum:<br/>
First implementation.
</li>
</ul>
</html>"));
        end ControllerEconomizer;

        model ControllerHeatingFan "Controller for heating and cooling"
          extends Modelica.Blocks.Icons.Block;

          parameter Real kHea(min=Modelica.Constants.small) = 1
            "Gain of heating controller"
            annotation(Dialog(group="Control gain"));

          parameter Real kFan(min=Modelica.Constants.small) = 1
            "Gain of controller for fan"
            annotation(Dialog(group="Control gain"));

          parameter Real minAirFlo(
            min=0,
            max=1,
            unit="1")
            "Minimum airflow rate of system";

          Modelica.Blocks.Interfaces.RealInput TSetRooCoo(
            final unit="K",
            displayUnit="degC") "Zone cooling setpoint"
            annotation (Placement(transformation(extent={{-140,-40},{-100,0}}),
                iconTransformation(extent={{-140,-40},{-100,0}})));
          Modelica.Blocks.Interfaces.RealInput TRoo(
            final unit="K",
            displayUnit="degC")
            "Zone temperature measurement"
            annotation (Placement(transformation(extent={{-140,-90},{-100,-50}}),
                iconTransformation(extent={{-140,-90},{-100,-50}})));
          Modelica.Blocks.Interfaces.RealInput TSetRooHea(
            final unit="K",
            displayUnit="degC") "Zone heating setpoint"
            annotation (Placement(transformation(extent={{-140,10},{-100,50}}),
                iconTransformation(extent={{-140,10},{-100,50}})));
          Modelica.Blocks.Interfaces.RealOutput yFan(final unit="1") "Control signal for fan"
            annotation (Placement(transformation(extent={{100,30},{120,50}})));
          Modelica.Blocks.Interfaces.RealOutput yHea(final unit="1")
            "Control signal for heating coil"
            annotation (Placement(transformation(extent={{100,-50},{120,-30}})));
          Buildings.Controls.Continuous.LimPID conHeaCoi(
            final k=kHea,
            controllerType=Modelica.Blocks.Types.SimpleController.PI,
            Ti=60,
            reset=Buildings.Types.Reset.Parameter)
                   "Controller for heating coil"
            annotation (Placement(transformation(extent={{-10,-50},{10,-30}})));
          Buildings.Controls.Continuous.LimPID conFan(
            final k=kFan,
            Ti=60,
            final yMax=1,
            final yMin=minAirFlo,
            controllerType=Modelica.Blocks.Types.SimpleController.PI,
            final reverseAction=true,
            reset=Buildings.Types.Reset.Parameter)
                                      "Controller for fan"
            annotation (Placement(transformation(extent={{20,30},{40,50}})));

          Modelica.Blocks.Interfaces.BooleanInput trigger
            "Resets the controller output when trigger becomes true"
            annotation (Placement(transformation(extent={{-140,60},{-100,100}})));
        equation
          connect(TSetRooHea, conHeaCoi.u_s)
            annotation (Line(points={{-120,30},{-60,30},{-60,-40},{-12,-40}},
                                                             color={0,0,127}));
          connect(TRoo, conHeaCoi.u_m) annotation (Line(points={{-120,-70},{0,
                  -70},{0,-52}},               color={0,0,127}));
          connect(conHeaCoi.y, yHea)
            annotation (Line(points={{11,-40},{60,-40},{110,-40}},
                                                         color={0,0,127}));
          connect(conFan.u_s, TSetRooCoo) annotation (Line(points={{18,40},{-40,
                  40},{-40,-20},{-120,-20}},             color={0,0,127}));
          connect(TRoo, conFan.u_m) annotation (Line(points={{-120,-70},{30,-70},
                  {30,28}},                             color={0,0,127}));

          connect(conFan.y, yFan)
            annotation (Line(points={{41,40},{41,40},{110,40}},
                                                           color={0,0,127}));
          connect(conFan.trigger, trigger) annotation (Line(points={{22,28},{22,
                  0},{-20,0},{-20,80},{-120,80}}, color={255,0,255}));
          connect(trigger, conHeaCoi.trigger) annotation (Line(points={{-120,80},
                  {-20,80},{-20,-80},{-8,-80},{-8,-52}}, color={255,0,255}));
          annotation (
          defaultComponentName="conHeaFan",
          Documentation(info="<html>
<p>
Controller for heating coil and fan speed.
</p>
</html>",         revisions="<html>
<ul>
<li>
June 21, 2017, by Michael Wetter:<br/>
Refactored implementation.
</li>
<li>
June 1, 2017, by David Blum:<br/>
First implementation.
</li>
</ul>
</html>"));
        end ControllerHeatingFan;

        model FanStatus "Fan status"

          Modelica.Blocks.Interfaces.RealInput TSetRooCoo(final unit="K", displayUnit="degC")
            "Zone cooling setpoint"
            annotation (Placement(transformation(extent={{-140,-50},{-100,-10}}),
                iconTransformation(extent={{-140,-50},{-100,-10}})));
          Modelica.Blocks.Interfaces.RealInput TRoo(final unit="K", displayUnit="degC")
            "Zone temperature measurement"
            annotation (Placement(transformation(extent={{-140,-100},{-100,-60}}),
                iconTransformation(extent={{-140,-100},{-100,-60}})));
          Modelica.Blocks.Interfaces.RealInput TSetRooHea(final unit="K", displayUnit="degC")
                                "Zone heating setpoint"
            annotation (Placement(transformation(extent={{-140,10},{-100,50}}),
                iconTransformation(extent={{-140,10},{-100,50}})));
          Buildings.Controls.OBC.CDL.Interfaces.BooleanInput uOcc
            "Current occupancy period, true if it is in occupant period"
            annotation (Placement(transformation(extent={{-140,60},{-100,100.5}}),
              iconTransformation(extent={{-180,40},{-100,120}})));
          Modelica.Blocks.MathBoolean.Or or1(nu=3)
            annotation (Placement(transformation(extent={{60,-10},{80,10}})));
          Modelica.Blocks.Interfaces.BooleanOutput fanOn "Fan on/off status"
            annotation (Placement(transformation(extent={{100,-10},{120,10}})));
          Modelica.Blocks.Math.Add dTCoo(k1=1, k2=-1)
            annotation (Placement(transformation(extent={{-60,20},{-40,40}})));
          Modelica.Blocks.Logical.Hysteresis hys1(uLow=-0.5, uHigh=0.5)
            annotation (Placement(transformation(extent={{-20,20},{0,40}})));
          Modelica.Blocks.Logical.Hysteresis hys2(uLow=-0.5, uHigh=0.5)
            annotation (Placement(transformation(extent={{-20,-50},{0,-30}})));
          Modelica.Blocks.Math.Add dTHea(k1=1, k2=-1)
            annotation (Placement(transformation(extent={{-60,-50},{-40,-30}})));
        equation
          connect(uOcc, or1.u[1]) annotation (Line(points={{-120,80.25},{40,
                  80.25},{40,4.66667},{60,4.66667}},
                                 color={255,0,255}));
          connect(or1.y, fanOn)
            annotation (Line(points={{81.5,0},{110,0}}, color={255,0,255}));
          connect(TRoo, dTCoo.u1) annotation (Line(points={{-120,-80},{-80,-80},{-80,36},
                  {-62,36}}, color={0,0,127}));
          connect(TSetRooCoo, dTCoo.u2) annotation (Line(points={{-120,-30},{-76,-30},{-76,
                  24},{-62,24}}, color={0,0,127}));
          connect(dTCoo.y, hys1.u)
            annotation (Line(points={{-39,30},{-22,30}}, color={0,0,127}));
          connect(dTHea.y, hys2.u)
            annotation (Line(points={{-39,-40},{-22,-40}}, color={0,0,127}));
          connect(TSetRooHea, dTHea.u1) annotation (Line(points={{-120,30},{-84,30},{-84,
                  -34},{-62,-34}}, color={0,0,127}));
          connect(TRoo, dTHea.u2) annotation (Line(points={{-120,-80},{-80,-80},{-80,-46},
                  {-62,-46}}, color={0,0,127}));
          connect(hys1.y, or1.u[2]) annotation (Line(points={{1,30},{36,30},{36,
                  0},{60,0}}, color={255,0,255}));
          connect(hys2.y, or1.u[3]) annotation (Line(points={{1,-40},{36,-40},{
                  36,-4.66667},{60,-4.66667}}, color={255,0,255}));
          annotation (Icon(coordinateSystem(preserveAspectRatio=false), graphics={
                                        Rectangle(
                extent={{-100,-100},{100,100}},
                lineColor={0,0,127},
                fillColor={255,255,255},
                fillPattern=FillPattern.Solid), Text(
                extent={{-150,150},{150,110}},
                textString="%name",
                lineColor={0,0,255})}), Diagram(coordinateSystem(preserveAspectRatio=false)));
        end FanStatus;
      annotation (preferredView="info", Documentation(info="<html>
<p>
This package contains base classes that are used to construct the models in
<a href=\"modelica://Buildings.Air.Systems.SingleZone.VAV\">
Buildings.Air.Systems.SingleZone.VAV</a>.
</p>
</html>"));
      end BaseClasses;
    annotation (preferredView="info", Documentation(info="<html>
<p>
VAV system model that serves a single thermal zone.
</p>
</html>"));
    end Control;
  end BaseClasses;
  annotation (uses(Modelica(version="3.2.3"),
      Buildings(version="7.0.0"),
      IBPSA(version="3.0.0"),
      FaultInjection(version="1.0.0"),
      ModelicaServices(version="3.2.3")),
    version="1");
end SingleZoneVAV;
