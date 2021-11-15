within FiveZoneVAV.VAVReheat.BaseClasses;
block BandDeviationSumTest
  extends Modelica.Icons.Example;

  FiveZoneVAV.VAVReheat.BaseClasses.BandDeviationSum bandDevSum(uppThreshold=26
         + 273.15, lowThreshold=22 + 273.15)
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
