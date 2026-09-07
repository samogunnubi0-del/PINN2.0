const PLAYLIST = 'PLO1y6V1SXjjNSSOZvV3PcFu4B1S8nfXBM';
const LS = {
  week: 'calc1a_class_week',
  streak: 'calc1a_streak',
  sessions: 'calc1a_sessions',
  last: 'calc1a_last_checkin',
  cp: 'calc1a_cp_done',
  formulas: 'calc1a_formula_srs',
  lectures: 'calc1a_lectures_done'
};

const KHAN = {
  calc1: 'https://www.khanacademy.org/math/calculus-1',
  limits: 'https://www.khanacademy.org/math/calculus-1/cs1-limits-and-continuity',
  limitLaws: 'https://www.khanacademy.org/math/calculus-1/cs1-limits-and-continuity/cs1-strategy-in-finding-limits/v/limit-example-1',
  derivatives: 'https://www.khanacademy.org/math/calculus-1/cs1-derivatives-definition-and-basic-rules',
  chain: 'https://www.khanacademy.org/math/calculus-1/cs1-differentiation-composite-implicit-inverse',
  apps: 'https://www.khanacademy.org/math/calculus-1/cs1-applying-derivatives-to-analyze-functions',
  related: 'https://www.khanacademy.org/math/calculus-1/cs1-contextual-applications-of-differentiation',
  integrals: 'https://www.khanacademy.org/math/calculus-1/cs1-integration-and-accumulation-of-change',
  usub: 'https://www.khanacademy.org/math/calculus-1/cs1-integration-and-accumulation-of-change/cs1-integration-with-u-substitution/v/u-substitution',
  diffeq: 'https://www.khanacademy.org/math/calculus-1/cs1-differential-equations',
  area: 'https://www.khanacademy.org/math/calculus-1/cs1-applications-of-integration',
  geometry: 'https://www.khanacademy.org/math/geometry/hs-geo-analytic-geometry',
  algebra: 'https://www.khanacademy.org/math/algebra2',
  trig: 'https://www.khanacademy.org/math/trigonometry'
};

const UNITS = [
  {id:'all', label:'All'},
  {id:'study', label:'How to study'},
  {id:'algebra', label:'Algebra + geometry'},
  {id:'limits', label:'Limits'},
  {id:'derivatives', label:'Derivatives'},
  {id:'applications', label:'Applications'},
  {id:'integrals', label:'Integrals'},
  {id:'exams', label:'Exam review'}
];

function Q(q, options, correct, exp){
  return {question:q, options:options, correct:correct, explanation:exp};
}

const UNIT_QUIZ = {
  study: [
    Q('What daily loop actually builds a Calc 1 A?', ['Read the chapter twice and highlight formulas','Watch a chunk, hide it, then do problems from memory','Copy the solution manual line by line','Only watch lectures the night before the test'], 1, 'Memory is built by retrieval. Watching without doing problems is entertainment.'),
    Q('Why do people fail Calc 1 even when they understood the lecture?', ['The ideas are too abstract for community college','They cannot remember algebra moves under time','The professor uses a secret textbook','Calculators are banned'], 1, 'The concepts are two ideas: slope and area. The points die on algebra.'),
    Q('How far ahead of class should you be?', ['Same day is enough','One week ahead so lecture is review','Finish the whole book in four days','Skip class once you watch YouTube'], 1, 'One week ahead is the A move. Lecture becomes a second pass, not first contact.')
  ],
  algebra: [
    Q('Rewrite 1/√x so a derivative rule can eat it.', ['x^{1/2}','x^{-1/2}','-x^{1/2}','1/x^{2}'], 1, 'Root in the denominator is a negative fractional power: 1/√x = x^{-1/2}.'),
    Q('x² − 9 factors as', ['(x-9)(x+1)','(x-3)(x+3)','(x-9)²','irreducible'], 1, 'Difference of squares. This is the #1 save on a 0/0 limit.'),
    Q('Distance between (x1,y1) and (x2,y2) is', ['|x2-x1|+|y2-y1|','√((x2-x1)²+(y2-y1)²)','((x2-x1)+(y2-y1))/2','(y2-y1)/(x2-x1)'], 1, 'Pythagorean theorem on the coordinate plane. That is analytic geometry.')
  ],
  limits: [
    Q('lim x→a f(x) exists only if', ['f(a) is defined','left-hand limit equals right-hand limit','the graph has no hole','the function is a polynomial'], 1, 'The value at a does not have to exist. Left must match right.'),
    Q('A limit that becomes 0/0 after plugging in means', ['the answer is 0','the limit does not exist','algebra: factor, conjugate, or cancel','write DNE and stop'], 2, '0/0 is a signal, not an answer. Fix the expression, then plug in again.'),
    Q('lim x→∞ of (3x²+1)/(5x²−7) is', ['0','∞','3/5','3x/5'], 2, 'Same degree: ratio of leading coefficients. Horizontal asymptote y=3/5.')
  ],
  derivatives: [
    Q('f′(a) is', ['the average slope from 0 to a','the slope of the tangent at x=a','the area under f','f(a+1)−f(a)'], 1, 'Instantaneous slope. Speedometer, not odometer.'),
    Q('Product rule for (uv)′ is', ['u′v′','u′v+uv′','(u′v−uv′)/v²','u′(v)'], 1, 'Keep one, differentiate the other, add. Both pieces stay in the picture.'),
    Q('Chain rule for f(g(x)) is', ['f′(x)g′(x)','f′(g(x))·g′(x)','f(g′(x))','g′(f(x))'], 1, 'Outside derivative, leave the inside, times inside derivative.')
  ],
  applications: [
    Q('A critical number of f is where', ['f(x)=0','f′(x)=0 or f′ DNE, in the domain','f″(x)=0','the graph crosses the x-axis'], 1, 'Candidates for max/min. Then test them.'),
    Q('If f′>0 on an interval, f is', ['concave up','increasing','at a maximum','undefined'], 1, 'First derivative is the slope. Positive slope means the graph is climbing.'),
    Q('Related rates first step is', ['differentiate immediately','draw, name variables, write the geometry equation','plug in the numbers you know','guess the triangle'], 1, 'Picture, equation, differentiate with respect to t, then plug numbers.')
  ],
  integrals: [
    Q('∫ x^n dx for n≠−1 is', ['n x^{n-1}+C','x^{n+1}/(n+1)+C','n x^{n+1}+C','x^n/n+C'], 1, 'Power rule backwards: add 1 to the power, divide by the new power, plus C.'),
    Q('The Fundamental Theorem says', ['derivatives and integrals are unrelated','∫_a^b f′(x) dx = f(b)−f(a)','every limit is a derivative','area is always positive'], 1, 'Net change: add up the rate from a to b and you get total change of f.'),
    Q('u-sub is for integrals that look like', ['a product of two unrelated functions','an inside function whose derivative is sitting next to it','any definite integral','only trig integrals'], 1, 'If you see g(x) and g′(x) hanging around, set u = inside.')
  ]
};

const VIDEO_DATA = [
  {id:'v01', youtubeId:'3dVaOT3pQqc', title:'The Best Way to Learn Calculus', mins:'10:11', unit:'study', kind:'study', req:true, why:'Start here. How to actually study this subject, not a content lecture.', topics:['Daily loop','Problems over rereading','Stay ahead'], khan:KHAN.calc1, quiz:'study'},
  {id:'v02', youtubeId:'pTdEWDdzVi0', title:'Mastering Calculus with the 1968 Classic', mins:'9:01', unit:'study', kind:'study', req:false, why:'Optional book talk. Skip if you already have a course textbook.', topics:['Textbooks'], khan:KHAN.calc1, quiz:'study'},
  {id:'v03', youtubeId:'_GWaUhGKNxo', title:'A wildly popular mid-century calculus book', mins:'8:19', unit:'study', kind:'study', req:false, why:'Optional. Do not let book shopping replace problems.', topics:['Textbooks'], khan:KHAN.calc1, quiz:'study'},
  {id:'v04', youtubeId:'MuZKQzAjz5A', title:'Super Thick Calculus Book', mins:'11:33', unit:'study', kind:'study', req:false, why:'Optional. Thick books are not the A. Worked problems are.', topics:['Textbooks'], khan:KHAN.calc1, quiz:'study'},
  {id:'v05', youtubeId:'LGBVa-fMfzc', title:'3 Ways to Learn Calculus on Your Own', mins:'9:18', unit:'study', kind:'study', req:false, why:'Optional methods video. Useful if you feel lost on how to self-study.', topics:['Self study'], khan:KHAN.calc1, quiz:'study'},
  {id:'v06', youtubeId:'EYMzXfNa8Mg', title:'Math from an M.I.T. Calculus Book', mins:'10:47', unit:'study', kind:'study', req:false, why:'Optional. Your community-college syllabus is enough for an A.', topics:['Textbooks'], khan:KHAN.calc1, quiz:'study'},
  {id:'v07', youtubeId:'D0xgEKLfaMM', title:'Learn Calculus Fast', mins:'9:49', unit:'study', kind:'study', req:false, why:'Optional pep. Speed comes from algebra fluency, not a hack.', topics:['Pacing'], khan:KHAN.algebra, quiz:'study'},
  {id:'v08', youtubeId:'QFPcDGq_AX4', title:'Calculus For Beginners: Get Started Here', mins:'9:59', unit:'study', kind:'study', req:true, why:'Second required study video. Then start Lecture 1.2.', topics:['Getting started'], khan:KHAN.calc1, quiz:'study'},
  {id:'v09', youtubeId:'m7dAhwe0ZTY', title:'The Easiest Calculus Book In The World', mins:'8:12', unit:'study', kind:'study', req:false, why:'Optional. If your class book is Larson or Stewart, stay with that.', topics:['Textbooks'], khan:KHAN.calc1, quiz:'study'},
  {id:'v10', youtubeId:'GffrHlVKsrM', title:'Self Study Calculus Like Richard Feynman', mins:'9:55', unit:'study', kind:'study', req:false, why:'Optional. Re-derive. Do not copy.', topics:['Notebook method'], khan:KHAN.calc1, quiz:'study'},
  {id:'k01', youtubeId:'', title:'Khan gap: analytic geometry (lines, distance, circles)', mins:'Khan', unit:'algebra', kind:'khan', req:true, why:'Class will use this without teaching it. Distance, midpoint, slope, circle equation.', topics:['Distance','Midpoint','Circle','Slope'], khan:KHAN.geometry, quiz:'algebra'},
  {id:'k02', youtubeId:'', title:'Khan gap: algebra that kills Calc 1 grades', mins:'Khan', unit:'algebra', kind:'khan', req:true, why:'Negative and fractional exponents, factoring. Do this before limits.', topics:['Exponents','Factoring'], khan:KHAN.algebra, quiz:'algebra'},
  {id:'v11', youtubeId:'0euyDNGEiZ4', title:'Lecture 1.2 Finding Limits Graphically and Numerically', mins:'35:42', unit:'limits', kind:'lecture', req:true, why:'What a limit even is, from a graph and a table. The first real lecture.', topics:['One-sided limits','Tables','DNE'], khan:KHAN.limits, quiz:'limits'},
  {id:'k03', youtubeId:'', title:'Khan gap: Lecture 1.3 limit laws (missing from playlist)', mins:'Khan', unit:'limits', kind:'khan', req:true, why:'The playlist jumps 1.2 to 1.4. Tests live in 1.3: plug in, factor, conjugate.', topics:['Limit laws','0/0','Conjugate'], khan:KHAN.limitLaws, quiz:'limits'},
  {id:'v12', youtubeId:'yOKO63P5FwA', title:'Lecture 1.4 Continuity and One-Sided Limits', mins:'1:25:30', unit:'limits', kind:'lecture', req:true, why:'Continuity, removable holes, IVT. Classic exam definitions.', topics:['Continuity','IVT','One-sided'], khan:KHAN.limits, quiz:'limits'},
  {id:'v13', youtubeId:'IszUqhcVq40', title:'Lecture 1.5 Infinite Limits', mins:'1:07:34', unit:'limits', kind:'lecture', req:true, why:'Vertical asymptotes. When the function blows up.', topics:['Vertical asymptotes','Infinite limits'], khan:KHAN.limits, quiz:'limits'},
  {id:'v14', youtubeId:'HhvN20798oE', title:'Lecture 2.1 The Derivative and the Tangent Line Problem', mins:'1:03:41', unit:'derivatives', kind:'lecture', req:true, why:'Definition of derivative. You must compute f′(a) from the limit at least once.', topics:['Definition','Tangent line'], khan:KHAN.derivatives, quiz:'derivatives'},
  {id:'v15', youtubeId:'3xW349GhtKU', title:'Lecture 2.2–2.4 Power, product, quotient, chain', mins:'2:23:29', unit:'derivatives', kind:'lecture', req:true, why:'The big rules video. Split it over two days. This is the heart of Calc 1.', topics:['Power','Product','Quotient','Chain'], khan:KHAN.chain, quiz:'derivatives'},
  {id:'v16', youtubeId:'fDuIvT20Pe8', title:'Lecture 2.5 Implicit Differentiation', mins:'57:07', unit:'derivatives', kind:'lecture', req:true, why:'When y is mixed with x. Circles, dy/dx, tangent lines.', topics:['Implicit','dy/dx'], khan:KHAN.chain, quiz:'derivatives'},
  {id:'v17', youtubeId:'Yv_YRRP-yt4', title:'Lecture 2.6 Related Rates', mins:'53:21', unit:'applications', kind:'lecture', req:true, why:'The word-problem unit people fear. It is a recipe, not talent.', topics:['Related rates','Ladder','Cone'], khan:KHAN.related, quiz:'applications'},
  {id:'v18', youtubeId:'EsjjQC1mqlg', title:'Lecture 3.1 Extrema on an Interval', mins:'51:36', unit:'applications', kind:'lecture', req:true, why:'Closed interval method: critical numbers plus endpoints.', topics:['Absolute extrema','Critical numbers'], khan:KHAN.apps, quiz:'applications'},
  {id:'v19', youtubeId:'goR8GB7lO9Q', title:'Lecture 3.2 Rolle and the Mean Value Theorem', mins:'1:07:29', unit:'applications', kind:'lecture', req:true, why:'Know the hypotheses. The picture: some tangent matches the secant.', topics:['MVT','Rolle'], khan:KHAN.apps, quiz:'applications'},
  {id:'v20', youtubeId:'hyBHhd4-ST4', title:'Lecture 3.3 First Derivative Test', mins:'39:17', unit:'applications', kind:'lecture', req:true, why:'Increasing/decreasing and local max/min from the sign of f′.', topics:['First derivative test'], khan:KHAN.apps, quiz:'applications'},
  {id:'v21', youtubeId:'50gv7lOP24o', title:'Lecture 3.4 Concavity and Second Derivative Test', mins:'1:27:51', unit:'applications', kind:'lecture', req:true, why:'f″ tells bending. Inflection points.', topics:['Concavity','Inflection'], khan:KHAN.apps, quiz:'applications'},
  {id:'v22', youtubeId:'1NbUtjnuyKA', title:'Exam 2 Review', mins:'1:08:45', unit:'exams', kind:'review', req:true, why:'Pause and do each problem before he finishes it.', topics:['Derivatives exam'], khan:KHAN.apps, quiz:'derivatives'},
  {id:'v23', youtubeId:'A3gBsJ_fW8U', title:'Lecture 3.5 Limits at Infinity', mins:'59:03', unit:'limits', kind:'lecture', req:true, why:'Horizontal asymptotes. Degree battle.', topics:['Limits at infinity'], khan:KHAN.limits, quiz:'limits'},
  {id:'v24', youtubeId:'40XRHIgbwpQ', title:'Lecture 3.6 A Summary of Curve Sketching', mins:'31:31', unit:'applications', kind:'lecture', req:true, why:'Domain, intercepts, asymptotes, f′, f″ on one graph.', topics:['Curve sketching'], khan:KHAN.apps, quiz:'applications'},
  {id:'v25', youtubeId:'ql81b8E7RiY', title:'Lecture 3.7 Optimization Problems', mins:'34:37', unit:'applications', kind:'lecture', req:true, why:'Write the constraint, reduce to one variable, then derivative.', topics:['Optimization'], khan:KHAN.related, quiz:'applications'},
  {id:'v26', youtubeId:'HOyDcHy7z04', title:'Lecture 3.8 Newton’s Method', mins:'41:52', unit:'applications', kind:'lecture', req:true, why:'Know the iteration formula cold.', topics:['Newton'], khan:KHAN.apps, quiz:'applications'},
  {id:'k04', youtubeId:'', title:'Khan gap: Chapter 4 integrals, Riemann sums, FTC', mins:'Khan', unit:'integrals', kind:'khan', req:true, why:'The playlist skips this. If your class is Larson or Stewart, this is a full exam unit.', topics:['Antiderivatives','FTC','Area'], khan:KHAN.integrals, quiz:'integrals'},
  {id:'k05', youtubeId:'', title:'Khan gap: u-substitution', mins:'Khan', unit:'integrals', kind:'khan', req:true, why:'Reverse chain rule. If you skip this, ln/exp integration later will feel random.', topics:['u-sub'], khan:KHAN.usub, quiz:'integrals'},
  {id:'v27', youtubeId:'foYjL9g2j5w', title:'Lecture 5.1 Derivative of ln x', mins:'40:05', unit:'derivatives', kind:'lecture', req:true, why:'d/dx[ln x]=1/x. Also logarithmic differentiation for ugly products.', topics:['ln','Log diff'], khan:KHAN.derivatives, quiz:'derivatives'},
  {id:'v28', youtubeId:'AFA7viB84eU', title:'Lecture 5.2 Integral of 1/x', mins:'37:54', unit:'integrals', kind:'lecture', req:true, why:'The missing n=−1 case: ∫ dx/x = ln|x|+C.', topics:['ln integral'], khan:KHAN.integrals, quiz:'integrals'},
  {id:'v29', youtubeId:'tiLoK7xsoRw', title:'Lecture 5.4 Exp functions, differentiate and integrate', mins:'1:15:54', unit:'derivatives', kind:'lecture', req:true, why:'e^x is its own derivative.', topics:['e^x'], khan:KHAN.derivatives, quiz:'derivatives'},
  {id:'v30', youtubeId:'wiKkk7YJrC4', title:'Lecture 6.2 Growth and decay (separable DE only)', mins:'42:52', unit:'integrals', kind:'lecture', req:true, why:'y′=ky. Separate, integrate, apply the initial condition.', topics:['Growth','Decay'], khan:KHAN.diffeq, quiz:'integrals'},
  {id:'k06', youtubeId:'', title:'Khan gap: area between curves', mins:'Khan', unit:'integrals', kind:'khan', req:true, why:'Often the last Calc 1 application. ∫ (top−bottom) dx.', topics:['Area between curves'], khan:KHAN.area, quiz:'integrals'},
  {id:'v31', youtubeId:'WVGGGTkEtKg', title:'Exam 4 Review', mins:'1:16:13', unit:'exams', kind:'review', req:true, why:'Pause after each prompt. Write your answer, then play.', topics:['Later-term exam'], khan:KHAN.calc1, quiz:'integrals'},
  {id:'v32', youtubeId:'K2WY7Hx0je0', title:'Final Exam Review', mins:'1:26:35', unit:'exams', kind:'review', req:true, why:'Treat this as a dress rehearsal. Misses go into formula gym the same day.', topics:['Final'], khan:KHAN.calc1, quiz:'integrals'}
];

const FORMULAS = [
  {id:'f01', unit:'algebra', name:'Negative exponent', tex:'x^{-n}=\\dfrac{1}{x^{n}}', use:'Whenever a derivative or limit has a fraction of a power, convert first.', remember:'A negative exponent is not a negative number. It is “flip me into the denominator.” 10^{-1} is a tenth, not −10.', trap:'Writing −x^n. That is a different (and usually wrong) thing.'},
  {id:'f02', unit:'algebra', name:'Fractional exponent', tex:'x^{m/n}=\\sqrt[n]{x^{m}}', use:'Power rule cannot see a radical until you rewrite it.', remember:'Bottom number is the root. Top number is the power. √x = x^{1/2}.', trap:'Leaving √x in the problem and then guessing the derivative.'},
  {id:'f03', unit:'algebra', name:'Difference of squares', tex:'a^{2}-b^{2}=(a-b)(a+b)', use:'The first tool when a limit is 0/0 and you see squares.', remember:'Two squares with a minus always split into plus and minus. x²−9=(x−3)(x+3).', trap:'Writing only (x−3) and dropping the plus partner.'},
  {id:'f04', unit:'algebra', name:'Quadratic formula', tex:'x=\\dfrac{-b\\pm\\sqrt{b^{2}-4ac}}{2a}', use:'Critical numbers when f′ is a quadratic you cannot factor fast.', remember:'Minus b, plus or minus the square root of b²−4ac, all over 2a.', trap:'Putting 2a only under the square root.'},
  {id:'f05', unit:'algebra', name:'Log rules', tex:'\\ln(ab)=\\ln a+\\ln b,\\quad \\ln\\!\\left(\\dfrac{a}{b}\\right)=\\ln a-\\ln b,\\quad \\ln(a^{k})=k\\ln a', use:'Expand before differentiating a log of a mess.', remember:'Logs turn multiply into add, divide into subtract, powers into a coefficient in front.', trap:'Writing ln(a+b)=ln a + ln b. Never. Logs do not split plus.'},
  {id:'f06', unit:'algebra', name:'Slope of a line', tex:'m=\\dfrac{y_{2}-y_{1}}{x_{2}-x_{1}}', use:'Analytic geometry and average rate of change.', remember:'Rise over run. How much y changes per one x. That sentence is already the derivative idea.', trap:'Putting x on top.'},
  {id:'f07', unit:'algebra', name:'Point-slope line', tex:'y-y_{1}=m(x-x_{1})', use:'Tangent line questions. Every Calc 1 test has one.', remember:'Start at the known point, then walk with slope m. Point-slope is the tangent-line formula.', trap:'Forcing y=mx+b and botching b when they already gave you a point.'},
  {id:'f08', unit:'algebra', name:'Distance formula', tex:'d=\\sqrt{(x_{2}-x_{1})^{2}+(y_{2}-y_{1})^{2}}', use:'Analytic geometry, related rates with a stretching segment.', remember:'It is Pythagoras. The legs are the horizontal and vertical gaps.', trap:'Forgetting the square root at the end.'},
  {id:'f09', unit:'algebra', name:'Circle', tex:'(x-h)^{2}+(y-k)^{2}=r^{2}', use:'Implicit differentiation and geometry constraints.', remember:'Every point is r away from the center (h,k). Square both sides of the distance formula and you get this.', trap:'Using x²+y²=r² when the center is not the origin.'},
  {id:'f10', unit:'algebra', name:'Pythagorean identity', tex:'\\sin^{2}\\theta+\\cos^{2}\\theta=1', use:'Trig limits, simplifying derivatives, related rates in a right triangle.', remember:'On the unit circle, x=cos, y=sin, and x²+y²=1. Same as the circle equation.', trap:'Mixing this up with 1+tan²=sec².'},
  {id:'f11', unit:'limits', name:'Two-sided limit exists', tex:'\\lim_{x\\to a}f(x)=L \\iff \\lim_{x\\to a^{-}}f(x)=\\lim_{x\\to a^{+}}f(x)=L', use:'Graph questions and piecewise functions.', remember:'Two hikers on a bridge. If they do not meet at the same height, the two-sided limit does not exist. A hole in the floor does not matter.', trap:'Saying DNE because there is a hole even though both sides meet.'},
  {id:'f12', unit:'limits', name:'0/0 is a signal', tex:'\\text{plug in }\\to 0/0 \\Rightarrow \\text{ algebra, then retry}', use:'Almost every algebraic limit on a test.', remember:'0/0 means the original form is lying. Factor, cancel, conjugate, then plug in again.', trap:'Writing 0 as the answer because the top is 0.'},
  {id:'f13', unit:'limits', name:'Limits at infinity', tex:'\\text{same degree }\\Rightarrow\\text{ ratio of leading coefficients}', use:'Horizontal asymptotes.', remember:'For big x, only the highest powers matter. Bigger bottom → 0. Bigger top → ±∞. Tie → ratio of leaders.', trap:'Trying to plug in ∞ like a number without comparing degrees.'},
  {id:'f14', unit:'derivatives', name:'Definition of derivative', tex:'f\'(x)=\\lim_{h\\to 0}\\dfrac{f(x+h)-f(x)}{h}', use:'When they say “use the definition.” Also the meaning of every later shortcut.', remember:'Average slope between x and x+h, then smash h to 0. That average becomes the tangent slope.', trap:'Forgetting to expand f(x+h) fully before the limit.'},
  {id:'f15', unit:'derivatives', name:'Power rule', tex:'\\dfrac{d}{dx}[x^{n}]=n x^{n-1}', use:'Almost every derivative in the course.', remember:'The power comes down in front and then gets tired, so it loses 1. x^4 → 4x^3. Rewrite radicals first.', trap:'Leaving the exponent the same.'},
  {id:'f16', unit:'derivatives', name:'Product rule', tex:'(uv)\'=u\'v+uv\'', use:'A product of two x-things, not a constant times a function.', remember:'Keep the first, differentiate the second, plus keep the second, differentiate the first.', trap:'Writing u′v′ only.'},
  {id:'f17', unit:'derivatives', name:'Quotient rule', tex:'\\left(\\dfrac{u}{v}\\right)\'=\\dfrac{u\'v-uv\'}{v^{2}}', use:'Fractions of functions.', remember:'Low d-high minus high d-low, over low low. Say it out loud every time. The minus is why order matters.', trap:'Flipping the minus. That one sign error is an entire test point.'},
  {id:'f18', unit:'derivatives', name:'Chain rule', tex:'\\dfrac{d}{dx}f(g(x))=f\'(g(x))\\,g\'(x)', use:'A function inside another function.', remember:'Peel the onion: derivative of the outside, leave the inside alone, then multiply by the inside’s derivative.', trap:'Forgetting to multiply by the inside derivative. The most common Calc 1 error after algebra.'},
  {id:'f19', unit:'derivatives', name:'Trig derivatives', tex:'(\\sin x)\'=\\cos x,\\quad (\\cos x)\'=-\\sin x', use:'Any sine/cosine problem. The co- functions pick up a minus.', remember:'Sine goes to cosine with no minus. Every co- function (cos, cot, csc) gets a minus.', trap:'Dropping the minus on cosine.'},
  {id:'f20', unit:'derivatives', name:'e^x and ln x', tex:'(e^{x})\'=e^{x},\\quad (\\ln x)\'=\\dfrac{1}{x}', use:'Exponential growth and any log derivative.', remember:'e^x is the function that is its own slope. ln is the inverse, so its slope is 1/x. Chain rule still applies.', trap:'Writing (ln x)′ = 1/ln x.'},
  {id:'f21', unit:'derivatives', name:'Implicit differentiation', tex:'\\text{diff both sides; } y^2 \\to 2y\\,y\'', use:'x²+y²=r² and any unsolved y.', remember:'y is a function of x, so differentiating y² uses the chain rule and produces y′. Then solve for y′.', trap:'Treating y as a constant so y² becomes 0.'},
  {id:'f22', unit:'applications', name:'Critical number', tex:'f\'(c)=0\\text{ or }f\'\\text{ DNE, }c\\text{ in domain}', use:'Every max/min problem.', remember:'Places the graph could turn: flat tangent or a corner. Candidates are not automatically extrema. Test them.', trap:'Reporting every critical number as a max without a test.'},
  {id:'f23', unit:'applications', name:'First derivative test', tex:'f\'\\text{ changes }+\\to - \\Rightarrow \\text{local max}', use:'Classifying local max/min.', remember:'Slope goes from climb to drop: peak. Drop to climb: valley. Read f′ like a hill profile.', trap:'Using only f′(c)=0 and calling it a max.'},
  {id:'f24', unit:'applications', name:'Second derivative test', tex:'f\'(c)=0,\\ f\'\'(c)>0 \\Rightarrow \\text{local min}', use:'Fast classification when f″ is easy.', remember:'f″>0 is a smile (concave up), so a flat point is a valley. f″<0 is a frown: peak.', trap:'If f″(c)=0 the test is silent. Switch to the first derivative test.'},
  {id:'f25', unit:'applications', name:'Mean Value Theorem', tex:'f\'(c)=\\dfrac{f(b)-f(a)}{b-a}', use:'Existence questions.', remember:'On a closed interval, some tangent matches the secant from a to b. Instantaneous speed equals average speed at least once.', trap:'Forgetting the hypotheses: continuous on [a,b], differentiable on (a,b).'},
  {id:'f26', unit:'applications', name:'Newton’s method', tex:'x_{n+1}=x_{n}-\\dfrac{f(x_{n})}{f\'(x_{n})}', use:'Approximate a root of f(x)=0.', remember:'Follow the tangent down to the x-axis. That intercept is the next guess. Repeat.', trap:'A bad first guess or a tiny f′ can throw you far away. Restart closer.'},
  {id:'f27', unit:'applications', name:'Related rates skeleton', tex:'\\text{geometry first, then } d/dt', use:'Ladder, balloon, shadow, leaking cone.', remember:'Draw. Label. Geometry equation with no rates yet. Then differentiate both sides with respect to t. Plug numbers last.', trap:'Plugging the given numbers into the geometry equation before differentiating.'},
  {id:'f28', unit:'integrals', name:'Antiderivative power rule', tex:'\\int x^{n}\\,dx=\\dfrac{x^{n+1}}{n+1}+C\\ (n\\neq -1)', use:'Indefinite integrals of polynomials.', remember:'Power rule in reverse: add 1 to the exponent, divide by that new number, plus C. Check by differentiating.', trap:'Forgetting +C, or using this formula on 1/x.'},
  {id:'f29', unit:'integrals', name:'FTC (net change)', tex:'\\int_{a}^{b}f\'(x)\\,dx=f(b)-f(a)', use:'Definite integrals and total-change word problems.', remember:'Adding up a rate from a to b recovers the net change of the original function. Odometer from speedometer.', trap:'Evaluating F(a)−F(b) instead of F(b)−F(a).'},
  {id:'f30', unit:'integrals', name:'u-substitution', tex:'u=g(x),\\quad du=g\'(x)\\,dx', use:'Integrals that are chain rule backwards.', remember:'If you see an inside function and its derivative sitting next to it, name the inside u.', trap:'Forgetting to convert dx to du, or forgetting to switch bounds on a definite integral.'},
  {id:'f31', unit:'integrals', name:'Area between curves', tex:'A=\\int_{a}^{b}(f_{\\mathrm{top}}-f_{\\mathrm{bottom}})\\,dx', use:'Last-unit application problems.', remember:'Height of each skinny rectangle is top minus bottom.', trap:'Integrating f−g when g is actually on top on part of the interval. Split the integral.'},
  {id:'f32', unit:'integrals', name:'Exponential growth', tex:'y=y_{0}e^{kt}', use:'Population, decay (k<0), some cooling models.', remember:'The DE y′=ky says the rate is proportional to the amount. Solution is exponential. Find k from the data.', trap:'Using the formula without finding k from the given information.'}
];

const PROBLEMS = [
  {id:'p1', unit:'algebra', prompt:'Write 1/√(x³) as a power of x.', options:['x^{3/2}','x^{-3/2}','x^{-1/3}','-x^{3/2}'], correct:1, steps:'√(x³)=(x³)^{1/2}=x^{3/2}. Reciprocal flips the sign: x^{-3/2}.'},
  {id:'p2', unit:'algebra', prompt:'Distance from (1,2) to (4,6).', options:['5','7','√13','25'], correct:0, steps:'Δx=3, Δy=4. √(9+16)=5. 3-4-5 triangle.'},
  {id:'p3', unit:'algebra', prompt:'Circle, center (2,−1), radius 3.', options:['(x-2)²+(y+1)²=9','(x+2)²+(y-1)²=9','(x-2)²+(y+1)²=3','x²+y²=9'], correct:0, steps:'(x−h)²+(y−k)²=r² with h=2, k=−1, r²=9. y−(−1)=y+1.'},
  {id:'p4', unit:'limits', prompt:'lim x→3 of (x²−9)/(x−3)', options:['0','DNE','6','3'], correct:2, steps:'0/0. Factor (x−3)(x+3)/(x−3)=x+3 for x≠3. Plug in 3: 6. Hole, but the limit is 6.'},
  {id:'p5', unit:'limits', prompt:'lim x→0 of (√(x+4)−2)/x', options:['0','1/4','1/2','DNE'], correct:1, steps:'0/0. Multiply by conjugate √(x+4)+2. Top becomes x. Cancel: 1/(√(x+4)+2) → 1/4.'},
  {id:'p6', unit:'limits', prompt:'lim x→∞ of (4x³−1)/(2x³+5x)', options:['0','∞','2','4'], correct:2, steps:'Same degree 3. Ratio of leading coefficients 4/2=2.'},
  {id:'p7', unit:'derivatives', prompt:'d/dx[√x]', options:['1/(2√x)','2√x','√x/2','1/√x'], correct:0, steps:'√x=x^{1/2}. Power rule: (1/2)x^{-1/2}=1/(2√x).'},
  {id:'p8', unit:'derivatives', prompt:'d/dx[(x²+1)(x³)] by product rule', options:['5x⁴+3x²','6x⁵','x⁵+x³','2x+3x²'], correct:0, steps:'u=x²+1, v=x³. 2x·x³+(x²+1)·3x²=2x⁴+3x⁴+3x²=5x⁴+3x².'},
  {id:'p9', unit:'derivatives', prompt:'d/dx[x/(x+1)]', options:['1/(x+1)','1/(x+1)²','x/(x+1)²','−1/(x+1)²'], correct:1, steps:'Quotient: (1·(x+1)−x·1)/(x+1)²=1/(x+1)².'},
  {id:'p10', unit:'derivatives', prompt:'d/dx[sin(3x)]', options:['cos(3x)','3 cos(3x)','−3 cos(3x)','3 sin(3x)'], correct:1, steps:'Chain rule. Sine → cosine of the inside, times 3.'},
  {id:'p11', unit:'derivatives', prompt:'If x²+y²=25, dy/dx at (3,4) is', options:['3/4','−3/4','4/3','−4/3'], correct:1, steps:'2x+2y y′=0 → y′=−x/y. At (3,4): −3/4.'},
  {id:'p12', unit:'applications', prompt:'13-ft ladder, bottom 5 ft from wall, dx/dt=2 ft/s. dy/dt of the top?', options:['−5/6 ft/s','−5/2 ft/s','−2 ft/s','−12/5 ft/s'], correct:0, steps:'x²+y²=169. When x=5, y=12. 2x x′+2y y′=0 → y′=−(x/y)x′=−(5/12)·2=−5/6 ft/s (top dropping).'},
  {id:'p13', unit:'applications', prompt:'f(x)=x³−3x on [−2,2]. Absolute maximum value?', options:['2','6','4','0'], correct:1, steps:'f′=3x²−3=0 at x=±1. f(−2)=−2, f(−1)=2, f(1)=−2, f(2)=6. Absolute max is 6 at the endpoint x=2.'},
  {id:'p14', unit:'applications', prompt:'f(x)=x³−3x. Local maximum occurs at', options:['x=1','x=−1','x=0','x=2'], correct:1, steps:'f′=3(x−1)(x+1). Sign of f′: + on (−∞,−1), − on (−1,1). Change + to − at x=−1: local max.'},
  {id:'p15', unit:'integrals', prompt:'∫ 3x² dx', options:['x³+C','3x³+C','6x+C','x²+C'], correct:0, steps:'Power rule backwards: 3·x³/3 + C = x³+C. Differentiate to check: 3x².'},
  {id:'p16', unit:'integrals', prompt:'∫_0^2 (2x) dx', options:['2','4','8','0'], correct:1, steps:'Antiderivative x². F(2)−F(0)=4−0=4. Area of a triangle with base 2 and height 4 is also 4.'},
  {id:'p17', unit:'integrals', prompt:'∫ 2x(x²+1)^3 dx. Best first move?', options:['Product rule','u=x²+1, du=2x dx','Expand everything','Quotient rule'], correct:1, steps:'Inside x²+1, and 2x dx is sitting there. That is u-sub.'},
  {id:'p18', unit:'integrals', prompt:'∫ dx/x equals', options:['x^{-2}+C','ln|x|+C','1/x²+C','e^x+C'], correct:1, steps:'n=−1 is the exception to the power rule. The antiderivative of 1/x is ln|x|+C.'}
];

const WEEKS = [
  {n:1, t:'Algebra + how to study', b:'Watch videos 1 and 8. Do Khan analytic geometry and algebra. Formula gym: exponents, distance, circle, slope.'},
  {n:2, t:'Limits from graphs', b:'Lecture 1.2. Khan limits intro. Formula: two-sided limit exists. Practice p4 style tables later this week.'},
  {n:3, t:'Limit laws (the missing 1.3)', b:'Khan gap k03 until 0/0 feels boring. Then Lecture 1.4 continuity and IVT.'},
  {n:4, t:'Infinite limits + derivative definition', b:'Lecture 1.5 then 2.1. Do at least one definition-of-derivative problem by hand.'},
  {n:5, t:'The big rules', b:'Lecture 2.2–2.4 over two days. Formula gym: power, product, quotient, chain. This week is the A.'},
  {n:6, t:'Implicit + related rates', b:'Lectures 2.5 and 2.6. Recipe card: draw, equation, d/dt, then numbers.'},
  {n:7, t:'Extrema, Rolle, MVT', b:'Lectures 3.1–3.2. Closed interval method until it is automatic: critical numbers plus endpoints.'},
  {n:8, t:'Shape of a graph', b:'Lectures 3.3–3.4. First and second derivative tests. Catch up any skipped algebra cards.'},
  {n:9, t:'Exam 2 week (stay ahead)', b:'Exam 2 review video. Then start 3.5 limits at infinity even if class is still on 3.3.'},
  {n:10, t:'Curve sketching + optimization', b:'Lectures 3.6–3.7. One full sketch and one fence/box problem every day.'},
  {n:11, t:'Newton, then integrals', b:'Lecture 3.8. Then start Khan Chapter 4 (k04). Do not wait for the playlist.'},
  {n:12, t:'FTC and u-sub', b:'Khan k04 and k05 until you can undo a chain-rule derivative in your sleep.'},
  {n:13, t:'ln and e^x', b:'Lectures 5.1, 5.2, 5.4. Connect them to the integral of 1/x.'},
  {n:14, t:'Growth/decay + area', b:'Lecture 6.2 and Khan area between curves. Formula: y=y0 e^{kt} and top minus bottom.'},
  {n:15, t:'Exam 4 + weak-point gym', b:'Exam 4 review. Any formula still in box 0 or 1 gets daily reps.'},
  {n:16, t:'Final', b:'Final review video. Light formula gym. Sleep. No new systems the last 48 hours.'}
];

const PEPS = [
  ['Formulas are sentences','Nobody is born knowing the product rule. Hide the symbols, say what they mean in English, then write them.'],
  ['Community college is not a lesser Calc 1','The tests still ask for the same tangent line and the same 0/0 limit. The A is practice, not the school’s zip code.'],
  ['Stay one week ahead','If class is on limits, you should already be differentiating. Lecture becomes review. Tests stop feeling like first contact.'],
  ['Algebra is the course in disguise','When a problem feels like “I cannot do calculus,” it is usually a fraction, a radical, or a factor. Fix that and the calculus is short.']
];

let classWeek = parseInt(localStorage.getItem(LS.week) || '1', 10);
let satStreak = parseInt(localStorage.getItem(LS.streak) || '0', 10);
let satSessions = parseInt(localStorage.getItem(LS.sessions) || '0', 10);
let lastCheckin = localStorage.getItem(LS.last) || '';
let cpDone = JSON.parse(localStorage.getItem(LS.cp) || '{}');
let formulaSRS = JSON.parse(localStorage.getItem(LS.formulas) || '{}');
let lecturesDone = JSON.parse(localStorage.getItem(LS.lectures) || '{}');
let ytPlayer = null, activeVideoIdx = -1, ytInterval = null, cpFired = {};
let cpCurrentAnswers = {}, cpTotalQ = 0, cpCorrect = 0;
let cpQuizBank = [];
let practiceQuizBank = { math: [] };
let videoFilter = 'all';
let formulaFilter = 'all';
let practiceFilter = 'algebra';
let formulaReveal = false;
let gymQueue = [];
let gymIndex = 0;

function tex(s, display){
  if(typeof katex === 'undefined') return s;
  try { return katex.renderToString(s, {throwOnError:false, displayMode:!!display}); }
  catch(e){ return s; }
}

function escHtml(s){
  return String(s==null?'':s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function todayKey(){
  const d = new Date();
  return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
}

function defaultClassWeek(){
  const start = new Date('2026-08-25T00:00:00');
  const w = Math.ceil((Date.now() - start.getTime()) / 86400000 / 7);
  return Math.min(16, Math.max(1, w));
}

document.addEventListener('DOMContentLoaded', () => {
  if(!localStorage.getItem(LS.week)) classWeek = defaultClassWeek();
  document.getElementById('live-date').textContent =
    new Date().toLocaleDateString('en-US',{weekday:'long',year:'numeric',month:'long',day:'numeric'});
  VIDEO_DATA.forEach(v => { if(!cpDone[v.id]) cpDone[v.id] = {25:false,50:false,75:false,100:false}; });
  FORMULAS.forEach(f => {
    if(!formulaSRS[f.id]) formulaSRS[f.id] = {box:0, due:todayKey(), seen:0, ok:0};
  });
  updateScoreBanner();
  updateStreakUI();
  buildFilters();
  buildVideoSelector();
  const first = VIDEO_DATA.findIndex(v => v.req && !isLectureDone(v.id));
  loadVideo(first >= 0 ? first : 0, true);
  checkCheckinState();
  renderPep();
  renderToday();
  renderWeek();
  renderFormulaGym();
  renderFormulaList();
  renderPractice();
  checkModelStatus();
});

function isLectureDone(id){
  const cps = cpDone[id] || {};
  return !!(cps[100] || lecturesDone[id]);
}

function targetWeek(){ return Math.min(16, classWeek + 1); }

function updateScoreBanner(){
  const t = targetWeek();
  document.getElementById('display-ahead').textContent = 'Class W'+classWeek+' · you W'+t;
  document.getElementById('ahead-mid').textContent = 'Your week '+t;
  document.getElementById('score-bar').style.width = Math.round((t/16)*100) + '%';
  const locked = FORMULAS.filter(f => (formulaSRS[f.id]||{}).box >= 3).length;
  document.getElementById('stat-formulas').textContent = locked;
  document.getElementById('stat-lectures').textContent = VIDEO_DATA.filter(v => v.req && isLectureDone(v.id)).length;
}

function openWeekModal(){
  document.getElementById('modal-score').value = classWeek;
  document.getElementById('score-modal').classList.add('open');
}
function closeModal(id){ document.getElementById(id).classList.remove('open'); }
function saveScore(){
  const v = parseInt(document.getElementById('modal-score').value,10);
  if(v>=1 && v<=16){
    classWeek = v;
    localStorage.setItem(LS.week, classWeek);
    updateScoreBanner();
    renderToday();
    renderWeek();
    closeModal('score-modal');
    showToast('Class week set to '+classWeek);
  }
}

function renderPep(){
  const i = new Date().getDate() % PEPS.length;
  document.getElementById('pep-title').textContent = PEPS[i][0];
  document.getElementById('pep-body').textContent = PEPS[i][1];
}

function nextRequiredVideo(){
  return VIDEO_DATA.find(v => v.req && !isLectureDone(v.id));
}

function dueFormulas(){
  const today = todayKey();
  return FORMULAS.filter(f => (formulaSRS[f.id].due || today) <= today);
}

function renderToday(){
  const due = dueFormulas();
  const next = nextRequiredVideo();
  const w = WEEKS[targetWeek()-1];
  const items = [
    'Formula gym: '+due.length+' card'+(due.length===1?'':'s')+' due. Hide the formula, say it in English, then rate it.',
    next ? ('Next required lesson: '+next.title+(next.kind==='khan'?' (Khan Academy gap)':'')+'.') : 'Required lectures are done. Use exam review and leftover formula cards.',
    'Stay-ahead target is week '+targetWeek()+': '+w.t+'. '+w.b,
    'Do 3 practice problems from today’s unit with the video off.'
  ];
  document.getElementById('today-mission').innerHTML = items.map((t,i)=>'<li><span class="ti-icon">'+(i+1)+'</span><span>'+escHtml(t)+'</span></li>').join('');
  const f = due[0] || FORMULAS[new Date().getDate() % FORMULAS.length];
  document.getElementById('today-formula').innerHTML = `
    <div class="formula-name">${escHtml(f.name)}</div>
    <div class="formula-use">${escHtml(f.use)}</div>
    <div class="formula-box">${tex(f.tex, true)}</div>
    <div class="remember-box"><div class="remember-label">How to remember</div>${escHtml(f.remember)}</div>`;
}

function renderWeek(){
  const t = targetWeek();
  document.getElementById('week-plan').innerHTML = WEEKS.map(w => `
    <div class="week-card ${w.n===classWeek?'current':''} ${w.n===t?'ahead':''}">
      <div style="font-size:.75rem;font-weight:800;color:var(--amber);margin-bottom:4px;">Week ${w.n}${w.n===classWeek?' · class is here':''}${w.n===t?' · you are here':''} · ${escHtml(w.t)}</div>
      <div style="font-size:.85rem;line-height:1.5;">${escHtml(w.b)}</div>
    </div>`).join('');
}

function updateCpStat(){
  let total = 0;
  VIDEO_DATA.forEach(v => { [25,50,75,100].forEach(cp => { if(cpDone[v.id] && cpDone[v.id][cp]) total++; }); });
  document.getElementById('stat-cp-done') && (document.getElementById('stat-cp-done').textContent = total);
  const ss = document.getElementById('ss-cp-total');
  if(ss) ss.textContent = total;
}

function switchTab(id){
  document.querySelectorAll('.tab-content').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.bn-btn').forEach(e=>e.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  const map={'t-today':'bn-today','t-sat':'bn-sat','t-formulas':'bn-formulas','t-practice':'bn-practice','t-tips':'bn-tips'};
  const bn = map[id]; if(bn) document.getElementById(bn).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b=>{ if(b.getAttribute('onclick')&&b.getAttribute('onclick').includes(id)) b.classList.add('active'); });
  if(id==='t-formulas') renderFormulaGym();
  if(id==='t-practice') renderPractice();
}

function checkCheckinState(){
  const today = new Date().toDateString();
  const btn = document.getElementById('checkin-btn');
  if(lastCheckin===today){ btn.textContent='Studied today. That counts.'; btn.disabled=true; btn.classList.add('done'); }
  else { btn.textContent='Mark today as studied'; btn.disabled=false; btn.classList.remove('done'); }
}
function doCheckin(){
  const today=new Date().toDateString();
  const yesterday=new Date(); yesterday.setDate(yesterday.getDate()-1);
  if(lastCheckin===today){ showToast('Already counted today',true); return; }
  if(lastCheckin!==yesterday.toDateString()&&satStreak>0) satStreak=0;
  satStreak++; satSessions++;
  lastCheckin=today;
  localStorage.setItem(LS.streak,satStreak);
  localStorage.setItem(LS.sessions,satSessions);
  localStorage.setItem(LS.last,lastCheckin);
  updateStreakUI(); checkCheckinState();
  showToast('Day '+satStreak+'. Keep the streak boring and consistent.');
}
function updateStreakUI(){
  document.getElementById('streak-count').textContent=satStreak;
  document.getElementById('ss-sessions').textContent=satSessions;
  document.getElementById('stat-sessions').textContent=satSessions;
  updateCpStat();
  updateScoreBanner();
}

function buildFilters(){
  const mk = (target, current, onclickName) => UNITS.map(u =>
    `<button type="button" class="cpill ${current===u.id?'active':''}" onclick="${onclickName}('${u.id}')">${escHtml(u.label)}</button>`
  ).join('');
  document.getElementById('unit-filters').innerHTML = mk('unit', videoFilter, 'setVideoFilter');
  document.getElementById('formula-filters').innerHTML = mk('formula', formulaFilter, 'setFormulaFilter');
  document.getElementById('practice-filters').innerHTML = UNITS.filter(u=>u.id!=='all'&&u.id!=='study'&&u.id!=='exams').concat([{id:'applications',label:'Applications'}]).filter((u,i,a)=>a.findIndex(x=>x.id===x.id)===i).map(u=>
    `<button type="button" class="cpill ${practiceFilter===u.id?'active':''}" onclick="setPracticeFilter('${u.id}')">${escHtml(u.label)}</button>`
  ).join('');
  const pf = ['algebra','limits','derivatives','applications','integrals'];
  document.getElementById('practice-filters').innerHTML = pf.map(id => {
    const label = UNITS.find(u=>u.id===id).label;
    return `<button type="button" class="cpill ${practiceFilter===id?'active':''}" onclick="setPracticeFilter('${id}')">${label}</button>`;
  }).join('');
}

function setVideoFilter(id){ videoFilter=id; buildFilters(); buildVideoSelector(); }
function setFormulaFilter(id){ formulaFilter=id; buildFilters(); renderFormulaList(); renderFormulaGym(); }
function setPracticeFilter(id){ practiceFilter=id; buildFilters(); renderPractice(); }

function filteredVideos(){
  return VIDEO_DATA.filter(v => videoFilter==='all' || v.unit===videoFilter);
}

function buildVideoSelector(){
  const el = document.getElementById('video-selector');
  el.innerHTML = filteredVideos().map((v) => {
    const i = VIDEO_DATA.indexOf(v);
    const cps = cpDone[v.id] || {};
    const dots = v.kind==='khan' ? '' : [25,50,75,100].map(cp =>
      `<div class="cp-dot ${cps[cp]?'done':''}" title="${cp}%">${cp}</div>`
    ).join('');
    const tag = v.kind==='khan' ? 'Khan gap' : (v.req ? 'Required' : 'Optional');
    return `<button type="button" class="vid-card" id="vc-${v.id}" onclick="loadVideo(${i})" aria-label="Open ${escHtml(v.title)}">
      <div class="vid-num">${tag} · ${escHtml(v.mins)}</div>
      <div class="vid-title">${escHtml(v.title)}</div>
      <div class="vid-creator">${v.kind==='khan'?'Khan Academy':'The Math Sorcerer'}</div>
      <div style="font-size:.78rem;color:var(--t2);margin-top:6px;line-height:1.4;">${escHtml(v.why)}</div>
      <div class="vid-checkpoints">${dots}</div>
    </button>`;
  }).join('');
}

function canUseYouTubeApi(){
  return location.protocol === 'http:' || location.protocol === 'https:';
}

window.onYouTubeIframeAPIReady = function(){
  if(activeVideoIdx >= 0 && canUseYouTubeApi()){
    const v = VIDEO_DATA[activeVideoIdx];
    if(v && v.youtubeId) connectYouTubePlayer(v);
  }
};

function connectYouTubePlayer(video){
  if(!video || !video.youtubeId || typeof YT === 'undefined' || !YT.Player || !document.getElementById('yt-player')) return;
  if(ytPlayer && typeof ytPlayer.getIframe === 'function'){
    try{ if(document.body.contains(ytPlayer.getIframe())) return; }catch(e){}
  }
  ytPlayer = null;
  try{
    ytPlayer = new YT.Player('yt-player', {
      events:{
        onReady: () => startTracking(video.id),
        onStateChange: (e) => onPlayerState(e, video.id),
        onError: () => {
          document.getElementById('yt-fallback').style.display = 'block';
          showToast('The player was blocked. The YouTube link is ready below.', true);
        }
      }
    });
  }catch(e){
    document.getElementById('yt-fallback').style.display = 'block';
  }
}

function loadVideo(idx, silent){
  const v = VIDEO_DATA[idx];
  if(!v) return;
  document.querySelectorAll('.vid-card').forEach(c=>c.classList.remove('active-vid'));
  const selectedCard = document.getElementById('vc-'+v.id);
  if(selectedCard){ selectedCard.classList.add('active-vid'); }
  activeVideoIdx = idx;
  cpFired = { ...(cpDone[v.id]||{}) };
  const wrap = document.getElementById('video-player-wrap');
  const fallback = document.getElementById('yt-fallback');
  const openLink = document.getElementById('yt-open-link');
  const watchUrl = v.youtubeId
    ? `https://www.youtube.com/watch?v=${v.youtubeId}&list=${PLAYLIST}`
    : v.khan;
  openLink.href = watchUrl;
  openLink.textContent = v.kind==='khan' ? 'Open this Khan Academy lesson' : `If needed, watch “${v.title}” on YouTube`;
  fallback.style.display = 'none';
  if(ytPlayer){ try{ ytPlayer.destroy(); }catch(e){} ytPlayer=null; }
  document.getElementById('khan-link').href = v.khan;
  document.getElementById('khan-link').textContent = v.kind==='khan' ? 'Work this unit on Khan Academy' : 'Same topic on Khan Academy';
  document.getElementById('vid-topic-info').style.display='block';
  document.getElementById('topics-list').innerHTML = v.topics.map(t=>`<span class="topic-tag">${escHtml(t)}</span>`).join('');

  if(v.kind==='khan' || !v.youtubeId){
    wrap.innerHTML = `<div class="video-placeholder video-local-help"><strong>This is a Khan Academy gap lesson</strong><span>The Math Sorcerer playlist does not teach this, and Calc 1 tests still do. Open Khan, then come back for the checkpoint quiz.</span><a class="open-local-btn" href="${v.khan}" target="_blank" rel="noopener" style="text-decoration:none;">Open Khan Academy</a></div>`;
    document.getElementById('cp-info').style.display='flex';
    document.getElementById('cp-bar-pct').textContent = 'Khan then quiz';
    document.getElementById('cp-bar-fill').style.width = '0%';
    fallback.style.display = 'block';
    if(!silent) showToast('Khan gap: '+v.title);
    return;
  }

  const apiBits = canUseYouTubeApi()
    ? `&enablejsapi=1&origin=${encodeURIComponent(location.origin)}&widget_referrer=${encodeURIComponent(location.href)}`
    : '';
  const embedUrl = `https://www.youtube.com/embed/${v.youtubeId}?rel=0&playsinline=1&list=${PLAYLIST}${apiBits}`;
  wrap.innerHTML = `<iframe id="yt-player" class="yt-frame" title="${escHtml(v.title)}" src="${embedUrl}" loading="eager" referrerpolicy="no-referrer-when-downgrade" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>`;
  if(canUseYouTubeApi()) connectYouTubePlayer(v);
  document.getElementById('cp-info').style.display='flex';
  document.getElementById('cp-bar-pct').textContent = canUseYouTubeApi() ? '0%' : 'use checkpoint quiz';
  document.getElementById('cp-bar-fill').style.width = '0%';
  updateCpUI(v.id);
  if(!silent) showToast('Playing: '+v.title);
}

function startTracking(videoId){
  clearInterval(ytInterval);
  ytInterval = setInterval(()=>{
    if(!ytPlayer || typeof ytPlayer.getCurrentTime !== 'function') return;
    try{
      if(ytPlayer.getPlayerState() !== YT.PlayerState.PLAYING) return;
      const cur = ytPlayer.getCurrentTime();
      const dur = ytPlayer.getDuration();
      if(!dur || dur <= 0) return;
      const pct = (cur/dur)*100;
      document.getElementById('cp-bar-fill').style.width = Math.min(100,pct)+'%';
      document.getElementById('cp-bar-pct').textContent = Math.round(pct)+'%';
      for(const cp of [25,50,75,100]){
        if(pct >= cp && !cpFired[cp]){
          cpFired[cp] = true;
          ytPlayer.pauseVideo();
          triggerCheckpoint(videoId, cp);
          break;
        }
      }
    }catch(e){}
  }, 1500);
}
function onPlayerState(e, videoId){
  if(e.data === YT.PlayerState.ENDED && !cpFired[100]){ cpFired[100]=true; triggerCheckpoint(videoId,100); }
}
function updateCpUI(videoId){
  buildVideoSelector();
  if(activeVideoIdx>=0){
    const activeCard = document.getElementById('vc-'+VIDEO_DATA[activeVideoIdx].id);
    if(activeCard) activeCard.classList.add('active-vid');
  }
  updateCpStat();
}
function manualCheckpoint(){
  if(activeVideoIdx<0){ showToast('Pick a lesson first',true); return; }
  const v = VIDEO_DATA[activeVideoIdx];
  const incomplete = [25,50,75,100].find(cp => !(cpDone[v.id]||{})[cp]);
  const cp = incomplete || 100;
  if(ytPlayer && ytPlayer.pauseVideo) ytPlayer.pauseVideo();
  triggerCheckpoint(v.id, cp);
}

function triggerCheckpoint(videoId, cpPct){
  const v = VIDEO_DATA.find(x=>x.id===videoId);
  document.getElementById('cp-modal-title').textContent = `${cpPct}% check · ${v.title}`;
  document.getElementById('cp-modal-sub').textContent = 'Three questions from this unit. Then you keep going.';
  document.getElementById('cp-modal').classList.add('open');
  const qs = UNIT_QUIZ[v.quiz] || UNIT_QUIZ.study;
  renderCheckpointQuiz({questions: qs}, v, cpPct);
}

function renderCheckpointQuiz(data, videoData, cpPct){
  const qs = (data.questions || []).slice(0,3);
  cpTotalQ = qs.length; cpCorrect = 0; cpCurrentAnswers = {};
  cpQuizBank = qs;
  document.getElementById('cp-quiz-area').innerHTML = qs.map((q,qi) => `
    <div class="quiz-q-card">
      <div class="quiz-q-num">Question ${qi+1} of ${cpTotalQ}</div>
      <div class="quiz-q-text">${escHtml(q.question)}</div>
      ${(q.options||[]).map((o,oi) => `<button type="button" class="quiz-opt" data-qi="${qi}" data-oi="${oi}">${escHtml(o)}</button>`).join('')}
      <div class="quiz-exp" id="cpq-exp-${qi}"></div>
    </div>`).join('') + `<div id="cp-summary" style="display:none;"></div>`;
}

function answerCpQ(qi, chosen, correct, explanation){
  [0,1,2,3].forEach(oi=>{
    const btn = document.querySelector(`#cp-quiz-area .quiz-opt[data-qi="${qi}"][data-oi="${oi}"]`);
    if(!btn) return;
    btn.disabled=true;
    if(oi===correct) btn.classList.add('correct');
    if(oi===chosen&&chosen!==correct) btn.classList.add('wrong');
  });
  const exp = document.getElementById(`cpq-exp-${qi}`);
  if(exp){ exp.textContent = (chosen===correct?'Nice. ':'Look at the story, not just the symbols. ') + explanation; exp.classList.add('show'); }
  if(!cpCurrentAnswers[qi]){
    cpCurrentAnswers[qi] = chosen===correct;
    if(chosen===correct) cpCorrect++;
  }
  if(Object.keys(cpCurrentAnswers).length >= cpTotalQ) setTimeout(showCpSummary, 400);
}

function showCpSummary(){
  const el = document.getElementById('cp-summary');
  if(!el) return;
  const msg = cpCorrect===cpTotalQ ? 'Locked in. Keep going.' : cpCorrect>=2 ? 'Good enough to continue. Misses go to formula gym later.' : 'Rewind two minutes, then try one similar problem on paper.';
  el.style.display='block';
  el.innerHTML=`<div class="quiz-summary"><div class="big-score">${cpCorrect}/${cpTotalQ}</div><div class="summary-label">${msg}</div><button class="resume-btn" onclick="completeCheckpoint()">Resume</button></div>`;
}

function completeCheckpoint(){
  if(activeVideoIdx>=0){
    const vid = VIDEO_DATA[activeVideoIdx];
    if(!cpDone[vid.id]) cpDone[vid.id] = {25:false,50:false,75:false,100:false};
    [25,50,75,100].forEach(cp=>{ if(cpFired[cp]) cpDone[vid.id][cp] = true; });
    if(vid.kind==='khan') cpDone[vid.id][100] = true;
    localStorage.setItem(LS.cp, JSON.stringify(cpDone));
    updateCpUI(vid.id);
  }
  closeCheckpointModal();
  showToast('Checkpoint done');
  setTimeout(()=>{ if(ytPlayer&&ytPlayer.playVideo) ytPlayer.playVideo(); }, 400);
}
function closeCheckpointModal(){
  document.getElementById('cp-modal').classList.remove('open');
  if(ytPlayer&&ytPlayer.playVideo) ytPlayer.playVideo();
}

function dueDatePlus(days){
  const d = new Date();
  d.setDate(d.getDate()+days);
  return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
}

function renderFormulaGym(){
  const today = todayKey();
  const pool = FORMULAS.filter(f => formulaFilter==='all' || f.unit===formulaFilter);
  gymQueue = pool.filter(f => (formulaSRS[f.id].due||today) <= today);
  if(gymQueue.length===0) gymQueue = pool.slice(0,1);
  gymIndex = Math.min(gymIndex, gymQueue.length-1);
  const dueN = FORMULAS.filter(f => (formulaSRS[f.id].due||today)<=today).length;
  document.getElementById('formula-due-label').textContent = dueN+' due today';
  const f = gymQueue[gymIndex];
  if(!f){ document.getElementById('formula-gym').innerHTML = '<p>All caught up.</p>'; return; }
  document.getElementById('formula-gym').innerHTML = `
    <div class="formula-stage">
      <div class="formula-name">${escHtml(f.name)}</div>
      <div class="formula-use">${escHtml(f.use)}</div>
      <div class="formula-box" id="formula-hidden">${formulaReveal ? tex(f.tex,true) : '<span style="color:var(--t2);font-weight:800;">Say it in English first. Then reveal.</span>'}</div>
      <button class="btn-sec" type="button" onclick="toggleFormulaReveal()">${formulaReveal?'Hide formula':'Reveal formula'}</button>
      <div class="remember-box"><div class="remember-label">How to remember</div>${escHtml(f.remember)}<div style="margin-top:8px;color:var(--t2);font-size:.86rem;"><strong>Trap:</strong> ${escHtml(f.trap)}</div></div>
      <div class="rate-row">
        <button class="rate-btn" onclick="rateFormula(0)">Blank</button>
        <button class="rate-btn" onclick="rateFormula(1)">Almost</button>
        <button class="rate-btn" onclick="rateFormula(2)">Locked</button>
      </div>
      <p style="margin-top:10px;font-size:.8rem;color:var(--t2);">${gymIndex+1} of ${gymQueue.length} in this set · box ${(formulaSRS[f.id].box||0)}</p>
    </div>`;
}

function toggleFormulaReveal(){
  formulaReveal = !formulaReveal;
  renderFormulaGym();
}

function rateFormula(kind){
  const f = gymQueue[gymIndex];
  if(!f) return;
  const s = formulaSRS[f.id];
  s.seen = (s.seen||0)+1;
  if(kind===0){ s.box = 0; s.due = dueDatePlus(1); }
  else if(kind===1){ s.box = Math.min(2, (s.box||0)); s.due = dueDatePlus(2); }
  else { s.ok = (s.ok||0)+1; s.box = Math.min(5, (s.box||0)+1); s.due = dueDatePlus([1,3,7,14,30,45][s.box] || 30); }
  localStorage.setItem(LS.formulas, JSON.stringify(formulaSRS));
  formulaReveal = false;
  gymIndex = (gymIndex+1) % Math.max(1, gymQueue.length);
  renderFormulaGym();
  renderFormulaList();
  updateScoreBanner();
  renderToday();
  showToast(kind===2 ? 'Locked. See you in a few days.' : kind===1 ? 'Again in 2 days.' : 'Back tomorrow. That is how memory is built.');
}

function renderFormulaList(){
  const pool = FORMULAS.filter(f => formulaFilter==='all' || f.unit===formulaFilter);
  document.getElementById('formula-list').innerHTML = pool.map(f => {
    const s = formulaSRS[f.id] || {box:0};
    return `<div class="gap-card">
      <div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;">
        <strong>${escHtml(f.name)}</strong>
        <span class="match-badge mh">box ${s.box||0}</span>
      </div>
      <div class="formula-box" style="margin:10px 0;">${tex(f.tex,true)}</div>
      <div style="font-size:.86rem;color:var(--t2);line-height:1.5;">${escHtml(f.remember)}</div>
    </div>`;
  }).join('');
}

function renderPractice(){
  const pool = PROBLEMS.filter(p => p.unit===practiceFilter);
  document.getElementById('practice-area').innerHTML = pool.map((p,i) => `
    <div class="quiz-q-card" data-pid="${p.id}">
      <div class="quiz-q-num">${p.unit} · ${i+1}/${pool.length}</div>
      <div class="problem-prompt">${escHtml(p.prompt)}</div>
      ${(p.options||[]).map((o,oi)=>`<button type="button" class="quiz-opt" data-practice="1" data-qi="${i}" data-oi="${oi}">${escHtml(o)}</button>`).join('')}
      <div class="quiz-exp">${escHtml(p.steps)}</div>
    </div>`).join('') || '<p>No drills in this unit yet.</p>';
}

async function checkModelStatus(){
  const box = document.getElementById('model-status');
  const text = document.getElementById('model-status-text');
  if(!box || !text) return;
  if(window.location.protocol === 'file:'){
    box.classList.add('ready');
    text.textContent = 'Exam drills on this page always work. No server needed.';
    return;
  }
  try{
    const response = await fetch('/api/status', {cache:'no-store'});
    const data = await response.json();
    if(data.configured){
      box.classList.add('ready');
      text.textContent = 'Extra trainer ready. Built-in drills still come first.';
    }else{
      box.classList.add('missing');
      text.textContent = 'Built-in drills always work. Extra trainer is optional.';
    }
  }catch(e){
    box.classList.add('missing');
    text.textContent = 'Built-in drills always work. Extra trainer is optional.';
  }
}

async function callCoach(userPrompt){
  if(window.location.protocol === 'file:') throw new Error('open-with-launcher');
  const response = await fetch('/api/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:userPrompt})
  });
  const data = await response.json().catch(()=>({}));
  if(!response.ok) throw new Error(data.error || 'model-request-failed');
  if(!data.content) throw new Error('empty-model-response');
  return data.content;
}

function parseJSON(text){
  text = String(text||'').replace(/```json\\s*/gi,'').replace(/```\\s*/g,'').trim();
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if(start!==-1 && end!==-1) text = text.substring(start, end+1);
  return JSON.parse(text);
}

function modelConnectionError(){
  return `<div class="model-error"><strong>The extra trainer is optional.</strong><br>Use the exam drills above. They do not need an internet model.</div>`;
}

async function generateMathAI(){
  const topic = document.getElementById('math-topic').value;
  const diff = document.getElementById('math-diff').value;
  const box = document.getElementById('math-ai-result');
  const btn = document.getElementById('math-gen-btn');
  btn.disabled = true;
  box.innerHTML = `<div class="ai-loading"><div class="spinner"></div><div class="ai-loading-text">Writing 3 ${diff} questions…</div></div>`;
  const prompt = `Write exactly 3 Calculus 1 exam questions on: ${topic}. Difficulty: ${diff}.
Community-college Calculus & Analytic Geometry 1. Show algebra. No trick 1550 puzzles.
JSON: {"questions":[{"question":"...","options":["A.","B.","C.","D."],"correct":0,"explanation":"plain steps"}]}`;
  try{
    const data = parseJSON(await callCoach(prompt));
    renderQuiz(box, data.questions, 'math');
  }catch(e){
    box.innerHTML = modelConnectionError();
  }
  btn.disabled = false;
}

function renderQuiz(box, questions, prefix){
  const qs = (questions||[]).slice(0,3);
  practiceQuizBank[prefix] = qs;
  box.innerHTML = qs.map((q,qi)=>`
    <div class="quiz-q-card" data-prefix="${prefix}" data-qi="${qi}">
      <div class="quiz-q-num">Extra ${qi+1}/3</div>
      <div class="quiz-q-text">${escHtml(q.question).replace(/\\n/g,'<br>')}</div>
      ${(q.options||[]).map((o,oi)=>`<button type="button" class="quiz-opt" data-practice="1" data-qi="${qi}" data-oi="${oi}">${escHtml(o)}</button>`).join('')}
      <div class="quiz-exp"></div>
    </div>`).join('');
}

function showToast(msg, warn){
  const t=document.getElementById('toast');
  t.textContent=msg;
  t.style.borderColor = warn? 'var(--rose)':'var(--border)';
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'), 2800);
}

document.addEventListener('click', e=>{
  if(e.target.id==='score-modal') closeModal('score-modal');
  if(e.target.id==='cp-modal') closeModal('cp-modal');
  const cpBtn = e.target.closest('#cp-quiz-area .quiz-opt');
  if(cpBtn && !cpBtn.disabled){
    const qi = +cpBtn.dataset.qi, oi = +cpBtn.dataset.oi;
    const q = cpQuizBank[qi];
    if(q) answerCpQ(qi, oi, q.correct, q.explanation||'');
    return;
  }
  const pBtn = e.target.closest('.quiz-opt[data-practice="1"]');
  if(pBtn && !pBtn.disabled){
    const card = pBtn.closest('.quiz-q-card');
    const qi = +pBtn.dataset.qi, oi = +pBtn.dataset.oi;
    if(card && card.dataset.pid){
      const pool = PROBLEMS.filter(p => p.unit===practiceFilter);
      const q = pool[qi];
      if(!q) return;
      card.querySelectorAll('.quiz-opt').forEach((b,i)=>{
        b.disabled = true;
        if(i===q.correct) b.classList.add('correct');
        if(i===oi && oi!==q.correct) b.classList.add('wrong');
      });
      const exp = card.querySelector('.quiz-exp');
      exp.textContent = (oi===q.correct?'Nice. ':'That is okay. ') + q.steps;
      exp.classList.add('show');
      return;
    }
    const prefix = card && card.dataset.prefix;
    const q = prefix && practiceQuizBank[prefix] && practiceQuizBank[prefix][qi];
    if(!q || !card) return;
    card.querySelectorAll('.quiz-opt').forEach((b,i)=>{
      b.disabled = true;
      if(i===q.correct) b.classList.add('correct');
      if(i===oi && oi!==q.correct) b.classList.add('wrong');
    });
    const exp = card.querySelector('.quiz-exp');
    exp.textContent = (oi===q.correct?'Nice. ':'That is okay. ') + (q.explanation||'');
    exp.classList.add('show');
  }
});

document.addEventListener('keydown', e=>{
  if(e.key === 'Escape'){
    closeModal('score-modal');
    closeModal('cp-modal');
  }
});
