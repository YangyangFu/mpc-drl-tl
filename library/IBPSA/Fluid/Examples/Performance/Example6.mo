within IBPSA.Fluid.Examples.Performance;
model Example6
  "Example 6 model of Modelica code that is inefficiently compiled into C-code"
  extends Modelica.Icons.Example;
  parameter Integer nCapacitors = 500;
  parameter Real R = 0.001;
   //annotation(Evaluate=true);
  parameter Real C = 1000;
   //annotation(Evaluate=true);

  Real[nCapacitors] T;
  Real[nCapacitors+1] Q_flow;
initial equation
  T = fill(273.15,nCapacitors);
equation
  Q_flow[1]=((273.15+sin(time))-T[1])/R;
  der(T[1])=(Q_flow[1]-Q_flow[2])/C;
  for i in 2:nCapacitors loop
    Q_flow[i] = (T[i-1] - T[i])/R;
    der(T[i])=(Q_flow[i]-Q_flow[i+1])/C;
  end for;
  Q_flow[nCapacitors+1]=0; //adiabatic

  annotation (Diagram(coordinateSystem(preserveAspectRatio=false, extent={{-120,
            -40},{40,60}}),    graphics={Text(
          extent={{-62,24},{-18,-4}},
          lineColor={0,0,255},
          textString="See code")}),
    experiment(
      Tolerance=1e-6, StopTime=100),
    Documentation(revisions="<html>
<ul>
<li>
April 11, 2016 by Michael Wetter:<br/>
Corrected wrong hyperlink in documentation for
<a href=\"https://github.com/ibpsa/modelica-ibpsa/issues/450\">issue 450</a>.
</li>
<li>
July 14, 2015, by Michael Wetter:<br/>
Revised documentation.
</li>
<li>
April 17, 2015, by Filip Jorissen:<br/>
First implementation.
</li>
</ul>
</html>", info="<html>
<p>This example, together with
<a href=\"modelica://IBPSA.Fluid.Examples.Performance.Example7\">
IBPSA.Fluid.Examples.Performance.Example7</a>,
illustrates the overhead
generated by divisions by parameters. See Jorissen et al. (2015) for a complementary discussion.
</p>
<p>
Running the following commands allows comparing the CPU times of the two models,
disregarding as much as possible the influence of the integrator:
</p>
<pre>
simulateModel(\"IBPSA.Fluid.Examples.PerformanceExamples.Example6\", stopTime=100, numberOfIntervals=1, method=\"Rkfix4\", fixedstepsize=0.001, resultFile=\"Example6\");
simulateModel(\"IBPSA.Fluid.Examples.PerformanceExamples.Example7\", stopTime=100, numberOfIntervals=1, method=\"Rkfix4\", fixedstepsize=0.001, resultFile=\"Example7\");
</pre>
<p>
Comparing the CPU times indicates a speed improvement of <i>56%</i>.
This difference almost disappears when adding <code>annotation(Evaluate=true)</code>
to <code>R</code> and <code>C</code>.
</p>
<p>
In <code>dsmodel.c</code> we find:
</p>
<pre>
DynamicsSection
W_[2] = divmacro(X_[0]-X_[1],\"T[1]-T[2]\",DP_[0],\"R\");
F_[0] = divmacro(W_[1]-W_[2],\"Q_flow[1]-Q_flow[2]\",DP_[1],\"C\");
</pre>
<p>
This suggests that the parameter division needs to be handled during
each function evaluation, probably causing the increased overhead.
</p>
<p>
The following command allows comparing the CPU times objectively.
</p>
<p>
<code>
simulateModel(\"IBPSA.Fluid.Examples.Performance.Example6\", stopTime=100, numberOfIntervals=1, method=\"Euler\", fixedstepsize=0.001, resultFile=\"Example6\");
</code>
</p>
<p>
See Jorissen et al. (2015) for a discussion.
</p>
<h4>References</h4>
<ul>
<li>
Filip Jorissen, Michael Wetter and Lieve Helsen.<br/>
Simulation speed analysis and improvements of Modelica
models for building energy simulation.<br/>
Submitted: 11th Modelica Conference. Paris, France. Sep. 2015.
</li>
</ul>
</html>"),
    __Dymola_Commands(file=
          "Resources/Scripts/Dymola/Fluid/Examples/Performance/Example6.mos"
        "Simulate and plot"));
end Example6;
