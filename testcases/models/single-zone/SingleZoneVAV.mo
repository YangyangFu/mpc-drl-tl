within ;
package SingleZoneVAV
  "This package contains models for the SingleZoneVAV testcase in BOPTEST."
  model TestCaseSupervisory
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

    Buildings.Air.Systems.SingleZone.VAV.ChillerDXHeatingEconomizerController con(
      minOAFra=0.2,
      kFan=4,
      kEco=4,
      kHea=4,
      TSupChi_nominal=TSupChi_nominal,
      TSetSupAir=286.15) "Controller"
      annotation (Placement(transformation(extent={{-100,-10},{-80,10}})));
    BaseClasses.ChillerDXHeatingEconomizer hvac(
      redeclare package MediumA = MediumA,
      redeclare package MediumW = MediumW,
      mAir_flow_nominal=0.75,
      etaHea_nominal=0.99,
      QHea_flow_nominal=7000,
      QCoo_flow_nominal=-7000,
      TSupChi_nominal=TSupChi_nominal)   "Single zone VAV system"
      annotation (Placement(transformation(extent={{-40,-20},{0,20}})));
    BaseClasses.Room  zon(
      redeclare package MediumA = MediumA,
        mAir_flow_nominal=0.75,
        lat=weaDat.lat) "Thermal envelope of single zone"
      annotation (Placement(transformation(extent={{40,-20},{80,20}})));
    Buildings.BoundaryConditions.WeatherData.ReaderTMY3 weaDat(
        computeWetBulbTemperature=false, filNam=
        Modelica.Utilities.Files.loadResource(
            "Resources/weatherdata/USA_CA_Riverside.Muni.AP.722869_TMY3.mos"))
      annotation (Placement(transformation(extent={{-160,120},{-140,140}})));
    Modelica.Blocks.Continuous.Integrator EFan "Total fan energy"
      annotation (Placement(transformation(extent={{40,-50},{60,-30}})));
    Modelica.Blocks.Continuous.Integrator EHea "Total heating energy"
      annotation (Placement(transformation(extent={{40,-80},{60,-60}})));
    Modelica.Blocks.Continuous.Integrator ECoo "Total cooling energy"
      annotation (Placement(transformation(extent={{40,-110},{60,-90}})));
    Modelica.Blocks.Math.MultiSum EHVAC(nu=4)  "Total HVAC energy"
      annotation (Placement(transformation(extent={{80,-70},{100,-50}})));
    Modelica.Blocks.Continuous.Integrator EPum "Total pump energy"
      annotation (Placement(transformation(extent={{40,-140},{60,-120}})));

    Buildings.BoundaryConditions.WeatherData.Bus weaBus "Weather data bus"
      annotation (Placement(transformation(extent={{-118,120},{-98,140}})));

    Modelica.Blocks.Sources.CombiTimeTable TSetRooHea(
      smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
      extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
      table=[0,THeaOff; 7*3600,THeaOff; 7*3600,THeaOn; 19*3600,THeaOn; 19*3600,
          THeaOff])   "Heating setpoint for room temperature"
      annotation (Placement(transformation(extent={{-180,20},{-160,40}})));
    Modelica.Blocks.Sources.CombiTimeTable TSetRooCoo(
      smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
      extrapolation=Modelica.Blocks.Types.Extrapolation.Periodic,
      table=[0,TCooOff; 7*3600,TCooOff; 7*3600,TCooOn; 19*3600,TCooOn; 19*3600,
          TCooOff])   "Cooling setpoint for room temperature"
      annotation (Placement(transformation(extent={{-180,-20},{-160,0}})));
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

    Modelica.Blocks.Interfaces.RealInput uFan "Fan control signal"
      annotation (Placement(transformation(extent={{-200,60},{-160,100}})));
    Modelica.Blocks.Interfaces.RealOutput TRoo
      "Connector of Real output signal"
      annotation (Placement(transformation(extent={{160,-10},{180,10}})));
    Modelica.Blocks.Interfaces.RealOutput CO2Roo
      "Connector of Real output signal"
      annotation (Placement(transformation(extent={{160,-40},{180,-20}})));
    Modelica.Blocks.Math.MultiSum PHVAC(nu=4)
      annotation (Placement(transformation(extent={{126,34},{138,46}})));
    Modelica.Blocks.Interfaces.RealOutput PTot
      annotation (Placement(transformation(extent={{160,30},{180,50}})));
    Modelica.Blocks.Interfaces.RealOutput TOut
      annotation (Placement(transformation(extent={{160,-70},{180,-50}})));
    Modelica.Blocks.Interfaces.RealOutput GHI
      annotation (Placement(transformation(extent={{160,-90},{180,-70}})));
  equation
    connect(weaDat.weaBus, weaBus) annotation (Line(
        points={{-140,130},{-108,130}},
        color={255,204,51},
        thickness=0.5), Text(
        string="%second",
        index=1,
        extent={{6,3},{6,3}}));

    connect(con.yHea, hvac.uHea) annotation (Line(points={{-79,6},{-40,6},{-56,6},
            {-56,12},{-42,12}},        color={0,0,127}));
    connect(con.yCooCoiVal, hvac.uCooVal) annotation (Line(points={{-79,0},{-54,0},
            {-54,5},{-42,5}},             color={0,0,127}));
    connect(con.yOutAirFra, hvac.uEco) annotation (Line(points={{-79,3},{-50,3},{
            -50,-2},{-42,-2}},             color={0,0,127}));
    connect(hvac.chiOn, con.chiOn) annotation (Line(points={{-42,-10},{-60,-10},{
            -60,-4},{-79,-4}},                                color={255,0,255}));
    connect(con.TSetSupChi, hvac.TSetChi) annotation (Line(points={{-79,-8},{-70,
            -8},{-70,-15},{-42,-15}},           color={0,0,127}));
    connect(con.TMix, hvac.TMix) annotation (Line(points={{-102,2},{-112,2},{
            -112,-40},{10,-40},{10,-4},{1,-4}},             color={0,0,127}));

    connect(hvac.supplyAir, zon.supplyAir) annotation (Line(points={{0,8},{10,8},
            {10,2},{40,2}},          color={0,127,255}));
    connect(hvac.returnAir, zon.returnAir) annotation (Line(points={{0,0},{6,0},{
            6,-2},{10,-2},{40,-2}},  color={0,127,255}));

    connect(con.TOut, weaBus.TDryBul) annotation (Line(points={{-102,-2},{-108,
            -2},{-108,130}},              color={0,0,127}));
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

    connect(hvac.PFan, EFan.u) annotation (Line(points={{1,18},{24,18},{24,-40},{
            38,-40}},  color={0,0,127}));
    connect(hvac.QHea_flow, EHea.u) annotation (Line(points={{1,16},{22,16},{22,
            -70},{38,-70}},
                       color={0,0,127}));
    connect(hvac.PCoo, ECoo.u) annotation (Line(points={{1,14},{20,14},{20,-100},
            {38,-100}},color={0,0,127}));
    connect(hvac.PPum, EPum.u) annotation (Line(points={{1,12},{18,12},{18,-130},{
            38,-130}},   color={0,0,127}));

    connect(EFan.y, EHVAC.u[1]) annotation (Line(points={{61,-40},{70,-40},{70,-54.75},
            {80,-54.75}},         color={0,0,127}));
    connect(EHea.y, EHVAC.u[2])
      annotation (Line(points={{61,-70},{64,-70},{64,-60},{66,-60},{66,-60},{80,-60},
            {80,-58.25}},                                      color={0,0,127}));
    connect(ECoo.y, EHVAC.u[3]) annotation (Line(points={{61,-100},{70,-100},{70,-61.75},
            {80,-61.75}},         color={0,0,127}));
    connect(EPum.y, EHVAC.u[4]) annotation (Line(points={{61,-130},{74,-130},{74,-65.25},
            {80,-65.25}},         color={0,0,127}));
    connect(TSetRooHea.y[1], oveTSetRooHea.u)
      annotation (Line(points={{-159,30},{-142,30}}, color={0,0,127}));
    connect(oveTSetRooHea.y, con.TSetRooHea) annotation (Line(points={{-119,30},
            {-116,30},{-116,10},{-102,10}}, color={0,0,127}));
    connect(TSetRooCoo.y[1], oveTSetRooCoo.u)
      annotation (Line(points={{-159,-10},{-142,-10}}, color={0,0,127}));
    connect(oveTSetRooCoo.y, con.TSetRooCoo) annotation (Line(points={{-119,-10},
            {-116,-10},{-116,6},{-102,6}}, color={0,0,127}));
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
    connect(zon.CO2, CO2RooAir.u) annotation (Line(points={{81,-4},{100,-4},{100,-30},
            {118,-30}}, color={0,0,127}));
    connect(TRooAir.y, TRoo)
      annotation (Line(points={{141,0},{170,0}}, color={0,0,127}));
    connect(CO2RooAir.y, CO2Roo)
      annotation (Line(points={{141,-30},{170,-30}}, color={0,0,127}));
    connect(PFan.y, PHVAC.u[1]) annotation (Line(points={{161,140},{174,140},{174,
            64},{94,64},{94,43.15},{126,43.15}}, color={0,0,127}));
    connect(PHea.y, PHVAC.u[2]) annotation (Line(points={{141,120},{170,120},{170,
            66},{92,66},{92,41.05},{126,41.05}}, color={0,0,127}));
    connect(PCoo.y, PHVAC.u[3]) annotation (Line(points={{161,100},{168,100},{168,
            68},{90,68},{90,38.95},{126,38.95}}, color={0,0,127}));
    connect(PPum.y, PHVAC.u[4]) annotation (Line(points={{141,80},{160,80},{160,68},
            {88,68},{88,36.85},{126,36.85}}, color={0,0,127}));
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
    connect(uFan, hvac.uFan) annotation (Line(points={{-180,80},{-60,80},{-60,
            18},{-42,18}}, color={0,0,127}));
    annotation (
      experiment(
        StopTime=504800,
        Interval=3600,
        Tolerance=1e-06),
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
  end TestCaseSupervisory;

  package BaseClasses "Base classes for test case"
    extends Modelica.Icons.BasesPackage;
    model Room "Room model for test case"
      extends Buildings.Air.Systems.SingleZone.VAV.Examples.BaseClasses.Room(roo(
            use_C_flow=true, nPorts=6), sinInf(use_C_in=true));
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

    model ChillerDXHeatingEconomizer "RTU model for test case"
      extends Buildings.Air.Systems.SingleZone.VAV.ChillerDXHeatingEconomizer(
          out(use_C_in=true));
      Modelica.Blocks.Sources.Constant conCO2Out(k=400e-6*Modelica.Media.IdealGases.Common.SingleGasesData.CO2.MM
            /Modelica.Media.IdealGases.Common.SingleGasesData.Air.MM)
        "Outside air CO2 concentration"
        annotation (Placement(transformation(extent={{-182,100},{-162,120}})));
    equation
      connect(conCO2Out.y, out.C_in[1]) annotation (Line(points={{-161,110},{
              -142,110},{-142,32}}, color={0,0,127}));
    end ChillerDXHeatingEconomizer;
  end BaseClasses;
  annotation (uses(Modelica(version="3.2.3"),
      Buildings(version="7.0.0"),
      IBPSA(version="3.0.0")),
    version="1");
end SingleZoneVAV;
