//
// Created by lisan on 2018/9/9.
//

#include <cstdio>
#include <cl.h>
#include <string.h>
#include "MapReduce.h"
#include "../Utils/OpenCLUtils.h"


const char * src = R"(
    __kernel void mapreduce(char16 pattern, __global char *text,
            int chars_per_item, __local int *local_result,
            __global int *global_result){

    uint8 mask;
    int16 a;
    int8 b;
    b = shuffle(a, mask);



    char16 text_vec, check_vec;
    local_result[0] = 0;
    local_result[1] = 0;
    local_result[2] = 0;
    local_result[3] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    int item_offset = get_global_id(0) * chars_per_item;

    for(int i = item_offset; i < item_offset + chars_per_item; ++i){
        text_vec = vload16(0, text + i);
        check_vec = text_vec == pattern;
        if(all(check_vec.s0123)){
            atomic_inc(local_result);
        }
        if(all(check_vec.s4567)){
            atomic_inc(local_result + 1);
        }
        if(all(check_vec.s89AB)){
            atomic_inc(local_result + 2);
        }
        if(all(check_vec.sCDEF)){
            atomic_inc(local_result + 3);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if(get_local_id(0) == 0){
        atomic_add(global_result, local_result[0]);
        atomic_add(global_result+1, local_result[1]);
        atomic_add(global_result+2, local_result[2]);
        atomic_add(global_result+3, local_result[3]);
    }

}
)";

//the text for search word
char * textmsg = R"(
21 Quotes That (If Applied) Change You Into a Better Person
As long as man has been alive, he has been collecting little sayings about how to live. We find them carved in the rock of the Temple of Apollo and etched as graffiti on the walls of Pompeii. They appear in the plays of Shakespeare, the commonplace book of H. P. Lovecraft, the collected proverbs of Erasmus, and the ceiling beams of Montaigne’s study. Today, they’re recorded on iPhones and in Evernote.

But whatever generation is doing it, whether they’re written by scribes in China or commoners in some European dungeon or simply passed along by a kindly grandfather, these little epigrams of life advice have taught essential lessons. How to respond to adversity. How to think about money. How to meditate on our mortality. How to have courage.

And they pack all this in in so few words. “What is an epigram?” Coleridge asked, “A dwarfish whole; Its body brevity, and wit its soul.” Epigrams are what Churchill was doing when he said: “To improve is to change, so to be perfect is to have changed often.” Or Balzac: “All happiness depends on courage and work.” Ah yes, epigrams are often funny too. That’s how we remember them. Napoleon: “Never interrupt an enemy making a mistake.” François de La Rochefoucauld: “We hardly find any persons of good sense save those who agree with us.” Voltaire: “A long dispute means that both parties are wrong.”

Below are some wonderful epigrams that span some 21 centuries and 3 continents. Each one is worth remembering, having queued in your brain for one of life’s crossroads or to drop at the perfect moment in conversation. Each will change and evolve with you as you evolve (Heraclitus: “No man steps in the same river twice”) and yet each will remain strong and unyielding no matter how much you may one day try to wiggle out and away from them.

Fundamentally, each one will teach you how to be a better person. If you let them.

“We must all either wear out or rust out, every one of us. My choice is to wear out.” — Theodore Roosevelt
At the beginning of his life, few would have predicted that Theodore Roosevelt even had a choice in the matter. He was sickly and fragile, doted on by worried parents. Then, a conversation with his father sent him driven, almost maniacally in the other direction. “I will make my body,” he said, when told that he would not go far in this world with a brilliant mind in a frail body. What followed was a montage of boxing, hiking, horseback riding, hunting, fishing, swimming, boldly charging enemy fire, and then a grueling work pace as one of the most prolific and admired presidents in American history. Again, this epigram was prophetic for Roosevelt, because at only 54 years old, his body began to wear out. An assassination attempt left a bullet lodged in his body and it hastened his rheumatoid arthritis. On his famous “River of Doubt” expedition he developed a tropical fever and the toxins from an infection in his leg left him nearly dead. Back in America he contracted a severe throat infection and was later diagnosed with inflammatory rheumatism, which temporarily confined him to a wheelchair (saying famously, “All right! I can work that way too!”) and then he died at age 60. But there is not a person on the planet who would say that he had not made a fair trade, that he had not worn his life well and not lived a full one in those 60 years.

“It’s not what happens to you, but how you react to it that matters.” — Epictetus
There is the story of the alcoholic father with two sons. One follows in his father’s footsteps and ends up struggling through life as a drunk, and the other becomes a successful, sober businessman. Each are asked: “Why are you the way you are?” The answer for both is the same: “Well, it’s because my father was an alcoholic.” The same event, the same childhood, two different outcomes. This is true for almost all situations — what happens to us is an objective reality, how we respond is a subjective choice. The Stoics — of which Epictetus was one — would say that we don’t control what happens to us, all we control are our thoughts and reactions to what happens to us. Remember that: You’re defined in this life not by your good luck or your bad luck, but your reaction to those strokes of fortune. Don’t let anyone tell you different.

“The best revenge is not to be like that.” — Marcus Aurelius
There is a proverb about revenge: Before setting out for a journey of revenge, dig two graves. Because revenge is so costly, because the pursuit of it often wears on the one who covets it. Marcus’s advice is easier and truer: How much better it feels to let it go, to leave the wrongdoer to their wrongdoing. And from what we know, Marcus Aurelius lived this advice. When Avidius Cassius, one of his most trusted generals rebelled and declared himself emperor, Marcus did not seek vengeance. Instead, he saw this as an opportunity to teach the Roman people and the Roman Senate about how to deal with civil strife in a compassionate, forgiving way. Indeed, when assassins struck Cassius down, Marcus supposedly wept. This is very different than the idea of “Living well being the best revenge” — it’s not about showing someone up or rubbing your success in their face. It’s that the person who wronged you is not happy, is not enjoying their life. Do not become like them. Reward yourself by being the opposite of them.

“There is good in everything, if only we look for it.” — Laura Ingalls Wilder
Laura Ingalls Wilder, author of the classic series Little House, lived this, facing some of the toughest and unwelcoming elements on the planet: harsh and unyielding soil, Indian territory, Kansas prairies, and the humid backwoods of Florida. Not afraid, not jaded — because she saw it all as an adventure. Everywhere was a chance to do something new, to persevere with cheery pioneer spirit whatever fate befell her and her husband. That isn’t to say she saw the world through delusional rose-colored glasses. Instead, she simply chose to see each situation for what it could be — accompanied by hard work and a little upbeat spirit. Others make the opposite choice. Remember: There is no good or bad without us, there is only perception. There is the event itself and the story we tell ourselves about what it means.

“Character is fate.” — Heraclitus
In the hiring process, most employers look at where someone went to school, what jobs they’ve held in the past. This is because past success can be an indicator of future successes. But is it always? There are plenty of people who were successful because of luck. Maybe they got into Oxford or Harvard because of their parents. And what about a young person who hasn’t had time to build a track record? Are they worthless? Of course not. This is why character is a far better measure of a man or woman. Not just for jobs, but for friendships, relationships, for everything. When you seek to advance your own position in life, character is the best lever — perhaps not in the short term, but certainly over the long term. And the same goes for the people you invite into your life.

“If you see fraud and do not say fraud, you are a fraud.” — Nicholas Nassim Taleb
A man shows up for work at a company where he knows that management is doing something wrong, something unethical. How does he respond? Can he cash his checks in good conscience because he isn’t the one running up the stock price, falsifying reports or lying to his co-workers? No. One cannot, as Budd Schulberg says in one of his novels, deal in filth without becoming the thing he touches. We should look up to a young man at Theranos as an example here. After discovering numerous problems at the health care startup, he was dismissed by his seniors and eventually contacted the authorities. Afterwards, not only was this young man repeatedly threatened, bullied, and attacked by Theranos, but his family had to consider selling their house to pay for the legal bills. His relationship with his grandfather — who sits on the Theranos board — is strained and perhaps irreparable. As Marcus Aurelius reminded himself, and us: “Just that you do the right thing. The rest doesn’t matter.” It’s an important reminder. Doing the right thing isn’t free. Doing the right thing might even cost you everything.

“Every man I meet is my master in some point, and in that I learn of him.” — Ralph Waldo Emerson
Everyone is better than you at something. This is a fact of life. Someone is better than you at making eye contact. Someone is better than you at quantum physics. Someone is better informed than you on geopolitics. Someone is better than you are at speaking kindly to someone they dislike. There are better gift-givers, name-rememberers, weight-lifters, temper-controllers, confidence-carriers, and friendship-makers. There is no one person who is the best at all these things, who doesn’t have room to improve in one or more of them. So if you can find the humility to accept this about yourself, what you will realize is that the world is one giant classroom. Go about your day with an openness and a joy about this fact. Look at every interaction as an opportunity to learn from and of the people you meet. You will be amazed at how quickly you grow, how much better you get.

“This is not your responsibility but it is your problem.” — Cheryl Strayed
It is not your responsibility to fill up a stranger’s gas tank, but when their car dies in front of you, blocking the road, it’s still your problem isn’t it? It is not your responsibility to negotiate peace treaties on behalf of your country, but when war breaks out and you’re drafted to fight in it? Guess whose problem it is? Yours. Life is like this. It has a way of dropping things into our lap — the consequences of an employee’s negligence, a spouse’s momentary lapse of judgement, a freak weather event — that were in no way our fault but by nature of being in our lap, our f*cking problem. So what are you going to do? Complain? Are you going to litigate this in a blogpost or an argument with God? Or are you just going to get to work solving it the best you can? Life is defined by how you answer that question. Cheryl Strayed is right. This thing might not be your responsibility but it is your problem. So accept it, deal with it, kick its ass.

“Waste no more time arguing what a good man should be. Be one.” — Marcus Aurelius
In Rome just as America, in the forum just as on Facebook, there was the temptation to replace action with argument. To philosophize instead of living philosophically. Today, in a society obsessed with content, outrage, and drama, it’s even easier to get lost in the echo chamber of the debate of what’s “better.” We can have endless discussions about what’s right and wrong. What should we do in this hypothetical situation or that one? How can we encourage other people to be better? (We can even debate the meaning of the above line: “What’s a man? What’s the definition of good? Why doesn’t it mention women?”) Of course, this is all a distraction. If you want to try to make the world a slightly better place, there’s a lot you can do. But only one thing guarantees an impact. Step away from the argument. Dig yourself out of the rubble. Stop wasting time with how things should be, would be, could be. Be that thing. (Here’s a cool poster of this quote).

“You are only entitled to the action, never to its fruits.” — Bhagavad Gita
In life, it’s a fact that: You will be unappreciated. You will be sabotaged. You will experience surprising failures. Your expectations will not be met. You will lose. You will fail. How do you carry on then? How do you take pride in yourself and your work? John Wooden’s advice to his players says it: Change the definition of success. “Success is peace of mind, which is a direct result of self satisfaction in knowing you made the effort to do your best to become the best that you are capable of becoming.” “Ambition,” Marcus Aurelius reminded himself, “means tying your well-being to what other people say or do . . . Sanity means tying it to your own actions.” Do your work. Do it well. Then “let go and let God.” That’s all there needs to be. Recognition and rewards — those are just extra.

“Self-sufficiency is the greatest of all wealth.” — Epicurus
A lot has been said of so-called “F*ck You Money.” The idea being that if one can earn enough, become rich and powerful enough, that suddenly no one can touch them and they can do whatever they want. What a mirage this is! How often the target seems to mysteriously move right as we approach it. It calls to mind the observation of David “DHH” Heinemeier Hansson who said that “beyond a specific amount, f*ck-you money can be a state of mind. One that you can acquire well in advance of the corresponding bank account. One that’s founded mostly on a personal confidence that even if most of the material trappings went away, you’d still be happier for standing your ground.” The truth is being your own man, being self-contained, having fewer needs, and better, resilient skills that allow you to thrive in any and all situations. That is real wealth and freedom. That’s what Emerson was talking about in his famous essay on self-reliance and it’s what Epicurus meant too.

“Tell me to what you pay attention and I will tell you who you are.” — Jose Ortega y Gasset
It was one of the great Stoics who said that if you live with a lame man, soon enough you will walk with a limp. My father told me something similar as a kid: “You become like your friends.” It is true not just with social influences but informational ones too: If you are addicted to the chatter of the news, you will soon find yourself worried, resentful, and perpetually outraged. If you consume nothing but escapist entertainment, you will find the real world around you harder and harder to deal with. If all you do is watch the markets and obsess over every fluctuation, your worldview will become defined by money and gains and losses. But if you drink from deep, philosophical wisdom? If you have regularly in your mind role models of restraint, sobriety, courage, and honor? Well, you will start to become these things too. Tell me who you spend time with, Goethe said, and I will tell you who you are. Tell me what you pay attention to, Gasset was saying, and I can tell you the same thing. Remember that the next time you feel your finger itching to pull up your Facebook feed.

“Better to trip with the feet than with the tongue.” — Zeno
You can always get up after you fall, but remember, what has been said can never be unsaid. Especially cruel and hurtful things.

“Space I can recover. Time, never.” — Napoleon Bonaparte
Lands can be reconquered, indeed in the course of a battle, a hill or a certain plain might trade hands several times. But missed opportunities? These can never be regained. Moments in time, in culture? They can never be re-made. One can never go back in time to prepare for what they should have prepared for, no one can ever get back critical seconds that were wasted out of fear or ego. Napoleon was brilliant at trading space for time: Sure, you can make these moves, provided you are giving me the time I need to drill my troops, or move them to where I want them to be. Yet in life, most of us are terrible at this. We trade an hour of our life here or afternoon there like it can be bought back with the few dollars we were paid for it. And it is only much much later, as they are on their deathbeds or when they are looking back on what might have been, that many people realize the awful truth of this quote. Don’t do that. Embrace it now.

“You never know who’s swimming naked until the tide goes out.” — Warren Buffett
The problem with comparing yourself to other people is you really never know anyone else’s situation. The co-worker with a nice car? It could be a dangerous and unsafe salvage with 100,000 miles. The friend who always seems to be traveling to far off places? They could be up to their eyeballs in credit card debt and about to get fired by their boss. Your neighbors’ marriage which makes you so insecure about your own? It could be a nightmare, a complete lie. People do a very good job pretending at things, and their well-maintained fronts are often covers for incredible risk and irresponsibility. You never know, Warren Buffett was saying, until things get bad. If you’re living the life you know to be right, if you are making good, solid decisions, don’t be swayed by what others are doing — whether that is taking the form of irrational exuberance or panicked pessimism. See the high flying lives of others as a cautionary tale — like Icarus with his wings — and not as an inspiration or a source of insecurity. Keep doing what you’re doing and don’t be caught swimming naked! Because the tide will go out. Prepare for it! (Premeditatio Malorum)

“Search others for their virtues, thyself for thy vices.” — Benjamin Franklin
Marcus Aurelius would say something similar: “Be tolerant with others and strict with yourself.” Why? For starters because the only person you control is yourself. It’s a complete waste of time to go around projecting strict standards on other people — ones they never agreed to follow in the first place — and then being aghast or feel wronged when they fall short. The other reason is you have no idea what other people are going or have been through. That person who seemed to rudely decline the invitation you so kindly offered? What if they were working hard to recommit themselves to their family and as much as they’d like to have coffee with you, are doing their best to spend more time with their loved ones? The point is: You have no idea. So give people the benefit of the doubt. Look for good in them, assume good in them, and let that good inspire your own actions.

“The world was not big enough for Alexander the Great, but a coffin was.” — Juvenal
Ah, the way that a good one liner can humble even the world’s greatest conqueror. Remember: we are all equals in death. It makes quick work of all of us, big and small. I carry a coin in my pocket to remember this: Memento Mori. What Juvenal reminds us is the same thing that Shakespeare spoke about in Hamlet:

“Imperious Caesar, dead and turned to clay,
Might stop a hole to keep the wind away.
O’ that that earth which kept the world in awe
Should patch a wall t’ expel the winder’s flaw!”
It doesn’t matter how famous you are, how powerful you are, how much you think you have left to do on this planet, the same thing happens to all of us, and it can happen when we least expect it. And then we will be wormfood and that’s the end of it.

“To improve is to change, so to be perfect is to have changed often.” — Winston Churchill
While this is probably not a Churchill original (he most likely borrowed from Cardinal Newman: “In a higher world it is otherwise, but here below to live is to change, and to be perfect is to have changed often”), Churchill certainly abided this in his life. He’d even quip about his constant change of political affiliation: “I said a lot of stupid things when I worked with the Conservative Party, and I left it because I did not want to go on saying stupid things.” As Cicero would say when attacked that he was changing his opinion: “If something strikes me as probable, I say it; and that is how, unlike everyone else, I remain a free agent.” There is nothing more impressive — intellectually or otherwise — than to change long held beliefs, opinions, and habits. The more you’ve changed, the better you probably are.

“Judge not, lest you be judged.” — Jesus
Not only here would Jesus call us on one of our worst tendencies but immediately also ask: “And why do you look at the speck in your brother’s eye, but do not consider the plank in your own eye?” This line is similar to what the Stoic philosopher Seneca, who historical sources suggest was born the same year as Jesus, would say: “You look at the pimples of others when you yourselves are covered with a mass of sores.” Waste no time judging and worrying about other people. You have plenty of problems to deal with in your own life. Chances are your own flaws are probably worse — and in any case, they are at least in your control. So do something about them.

“Time and patience are the strongest warriors.” — Leo Tolstoy
Tolstoy puts the above words in the mouth of Field Marshall Mikhail Kutuzov in War and Peace. In real life, Kutuzov gave Napoleon a painful lesson in the truth of the epigram over a long winter in Russia in 1812. Tolstoy would also say, “Everything comes in time to him who knows how to wait.” When it comes to accomplishing anything significant, you are required to exhibit patience and fortitude, so much patience, as much as you’d think you’d need boldness and courage.

“No one saves us but ourselves / No one can and no one may.” — Buddha
Will we wait for someone to save us, or will we listen to Marcus Aurelius’s empowering call to “get active in your own rescue — if you care for yourself at all — and do it while you can.”

Because at some point, we must put articles like this one aside and take action. No one can blow our nose for us. Another blog post isn’t the answer. The right choices and decisions are. Who knows how much time you have left, or what awaits us tomorrow? So get to it.

Like to Read?
I’ve created a list of 15 books you’ve never heard of that will alter your worldview and help you excel at your career.

Get the secret book list here!
)";


//search for string by the algorithm of MapReduce from Google
void mapReduceByOCL()
{
    cl_int err = 0;
    size_t group_cnt = 0;
    size_t group_size = 0;
    size_t global_item_cnt = 0;
    cl_device_id device = createDevice(CL_DEVICE_TYPE_GPU);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(group_cnt), &group_cnt, 0);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(group_size), &group_size, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_program program = buildProgram(context, device, src, strlen(src));

    char pattern[16] = "thatwithhavefro";
    pattern[15] = 'm';
    int result[4] = {0};
    cl_mem text_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, strlen(textmsg), textmsg, &err);
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(result), result, &err);
    checkCLError(err, "163");
    group_cnt = 10;     //because the textmsg is short,so, let items less
    group_size = 10;
    global_item_cnt = group_cnt * group_size;
    int chars_per_item = strlen(textmsg) / global_item_cnt + 1;
    LOGD( "chars_per_item %d %d %d %d", chars_per_item, group_cnt, group_size, strlen(textmsg));

    cl_kernel kernel = clCreateKernel(program, "mapreduce", &err);
    clSetKernelArg(kernel, 0, sizeof(pattern), pattern);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &text_buffer);
    clSetKernelArg(kernel, 2, sizeof(chars_per_item), &chars_per_item);
    clSetKernelArg(kernel, 3, sizeof(int) * 4, NULL);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &result_buffer);
    checkCLError(err, "173");


    cl_command_queue cmdqueue = clCreateCommandQueue(context, device,
                                                     CL_QUEUE_PROFILING_ENABLE, &err);
    checkCLError(err, "178");

    size_t offset = 0;
    err = clEnqueueNDRangeKernel(cmdqueue, kernel, 1, &offset, &global_item_cnt,
            &group_size, 0, NULL, NULL);
    checkCLError(err, "range");
    err = clEnqueueReadBuffer(cmdqueue, result_buffer, CL_TRUE, 0,
                        sizeof(result), &result, 0, NULL, NULL);
    checkCLError(err, "readbuffer");
    LOGD("RESULT :%d %d %d %d", result[0] ,result[1], result[2], result[3]);

    clReleaseMemObject(text_buffer);
    clReleaseMemObject(result_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdqueue);
    clReleaseContext(context);

    LOGD("mapreduce   end");


}