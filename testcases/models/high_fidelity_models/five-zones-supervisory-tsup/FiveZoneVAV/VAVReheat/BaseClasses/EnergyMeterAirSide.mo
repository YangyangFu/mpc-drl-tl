within FiveZoneVAV.VAVReheat.BaseClasses;
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
