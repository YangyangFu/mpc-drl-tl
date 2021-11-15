within FiveZoneVAV.VAVReheat.BaseClasses;
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
