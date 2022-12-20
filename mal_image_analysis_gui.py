import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import filedialog
from email.message import EmailMessage
import ssl
import smtplib
import customtkinter as ctk
import time
from objects.Structures import UnetParam
from objects.Analyzer import Analyzer
import javabridge
import bioformats


def run_through_gui(analysis_type, bioformat_imgs_path,
                    nuc_recognition_mode, mask_channel_name,nuc_area_min_pixels_num,
                    nuc_threshold, isWatershed, perinuclear_area, analysis_out_path):

    track_movement = True if analysis_type == 'tracing' else False

    # unet_model_path_63x = r"unet\models\CP_epoch198.pth"
    unet_model_path_63x = r"unet\models\CP_epoch72.pth"

    # unet_model_path_20x = r"unet\models\CP_epoch65_only20x_no-aug.pth" #old
    unet_model_path_20x = r"unet\models\CP_epoch172.pth" #new from Omar imgs

    # Unet training process characteristics:
    unet_model_scale = 1
    unet_img_size = (512, 512)
    unet_model_thrh = 0.5
    unet_parm = UnetParam(unet_model_path_63x, unet_model_path_20x, unet_model_scale, unet_model_thrh, unet_img_size)
    javabridge.start_vm(class_path=bioformats.JARS)

    start = time.time()
    analyser = Analyzer(bioformat_imgs_path, nuc_recognition_mode, nuc_threshold, unet_parm, nuc_area_min_pixels_num,
                        mask_channel_name, isWatershed, track_movement, trackEachFrame=False, perinuclearArea=perinuclear_area, analysis_out_path=analysis_out_path)
    analyser.run_analysis()
    end = time.time()
    print("Total time is: ")
    print(end - start)
    javabridge.kill_vm()


def send_email(subject, body, email_receiver):
    email_sender = 'mal5296963@gmail.com'
    email_password = 'jnyxcbwqxkmkyeic'
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())


def show_data_page():
    analysis_frame.destroy()
    data_frame.pack(expand=True)
    window_width = 650
    window_height = 600

    # get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # find the center point
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')


def run_analysis():

    print(f"Analysis type: {analysis_page.selected_type.get()}")
    analysis_type = analysis_page.selected_type.get() #'tracing' or 'intensity'

    print(f"Input folder: {data_page.input_folder.get()}")
    bioformat_imgs_path = data_page.input_folder.get()

    print(f"Output folder: {data_page.output_folder.get()}")
    output_folder = data_page.output_folder.get()

    print(f"Mask channel: {data_page.mask_channel.get()}")
    mask_channel_name = data_page.mask_channel.get()

    print(f"Nuc identification mode: {data_page.nuc_idnt_mode.get()}")
    nuc_recognition_mode = data_page.nuc_idnt_mode.get()

    print(f"Pixel Threshold: {data_page.thr.get()}, or as int {int(float(data_page.thr.get()))}")
    nuc_threshold = int(float(data_page.thr.get()))

    print(f"Minimal nucleus area: {data_page.min_nuc_area.get()}")
    nuc_area_min_pixels_num = data_page.min_nuc_area.get()

    print(f"Watershed: {data_page.separating_cells.get()}")
    isWatershed = True if data_page.separating_cells.get() == "true" else False

    print(f"Perinuclear area: {data_page.perinuclear_area.get()}")
    perinuclear_area = True if data_page.perinuclear_area.get() == "true" else False

    print(f"Email receiver: {data_page.email.get()}")
    email_receiver = data_page.email.get()

    if bioformat_imgs_path == "" or output_folder == "" or email_receiver == "" or nuc_area_min_pixels_num == "":
        showinfo(
            title='Information',
            message="Please fill all blanks to make the program run"
        )

    else:
        showinfo(
            title='Information',
            message="Analysis is about to start."
                    "You will be notified by email as soon as the results will be ready."
                    "Do you want to continue?"
            )
        try:
            root.destroy()
            run_through_gui(analysis_type, bioformat_imgs_path,
                            nuc_recognition_mode, mask_channel_name,
                            int(nuc_area_min_pixels_num), nuc_threshold,
                            isWatershed, perinuclear_area, output_folder)
            #Send congrats e-mail
            # TODO: Where email is actually written
            subject = "MAL: Results are ready"
            body = f"""
            Your data were processed. You decided to save the results in the following folder on your computer: 
            
            {output_folder}
            
            ...Results description...
            """
            send_email(subject, body, email_receiver)
            # showinfo(
            #     title='Information',
            #     message="Analysis is completed!"
            # )
        except Exception as e:
            #Send oops e-mail
            subject = "MAL: Oops error occurred"
            body = f"""
                        Unfortunately, we were not able to process your data. 
                        The following error occurred: {str(e)}
                        Please contact the developer if you will not be able to solve this problem
                        """
            send_email(subject, body, email_receiver)
            # showinfo(
            #     title='Information',
            #     message="Unfortunately, we were not able to process your data."
            # )


class AnalysisTypePage:
    def __init__(self, analysis_root):
        self.selected_type = tk.StringVar()
        analysis_types = (('Cells Tracking', 'tracing'),
                        ('Immunostained Image Analysis', 'intensity'))

        # label
        label = ctk.CTkLabel(master=analysis_root, text="Choose Type of Analysis: ")
        label.pack(fill='x', padx=10, pady=15)

        # Analysis type radio buttons
        self.selected_type.set('intensity')
        for type in analysis_types:
            r = ctk.CTkRadioButton(
                analysis_root,
                text=type[0],
                value=type[1],
                variable=self.selected_type
            )
            r.pack(fill='x', padx=50, pady=5)

        # Continue button
        self.continue_button = ctk.CTkButton(
            analysis_root,
            text="Continue")

        self.continue_button.pack(ipadx=5, ipady=5, expand=True, padx=0, pady=30)


class DataCollectionPage:
    def __init__(self, data_root):
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.email = tk.StringVar()
        self.mask_channel = tk.StringVar()
        self.nuc_idnt_mode = tk.StringVar()
        self.thr = tk.DoubleVar()
        self.min_nuc_area = tk.StringVar()
        self.separating_cells = tk.StringVar()
        self.perinuclear_area = tk.StringVar()

        # configure the grid for data root label
        data_root.columnconfigure(0, weight=1)
        data_root.columnconfigure(1, weight=2)

        # input folder path
        ctk.CTkLabel(master=data_root, text='Input folder', anchor='w').grid(column=0, row=1, sticky=tk.W, padx=15, pady=15)
        input_parent = ctk.CTkLabel(master=data_root)
        input_parent.grid(column=1, row=1, sticky=tk.EW)
        input_parent.columnconfigure(0, weight=10)
        input_parent.columnconfigure(1, weight=1)

        input_textbox = ctk.CTkEntry(master=input_parent, textvariable=self.input_folder)
        input_textbox.grid(row=0, column=0, sticky=tk.EW)
        button_br_1 = ctk.CTkButton(master=input_parent, text="...", command=self.input_button, width=30)
        button_br_1.grid(row=0, column=1, padx=5)

        # output folder path
        ctk.CTkLabel(master=data_root, text='Output folder', anchor='w').grid(column=0, row=2, sticky=tk.W, padx=15, pady=15)
        input_parent = ctk.CTkLabel(master=data_root)
        input_parent.grid(column=1, row=2, sticky=tk.EW)
        input_parent.columnconfigure(0, weight=10)
        input_parent.columnconfigure(1, weight=1)

        input_textbox = ctk.CTkEntry(master=input_parent, textvariable=self.output_folder)
        input_textbox.grid(row=0, column=0, sticky=tk.EW)
        button_br_2 = ctk.CTkButton(master=input_parent, text="...", command=self.output_button, width=30)
        button_br_2.grid(row=0, column=1, padx=5)

        # notification e-mail - just takes in the email address; it's not manipulated here
        ctk.CTkLabel(master=data_root, text='E-mail', anchor='w').grid(column=0, row=3, sticky=tk.W, padx=15, pady=15)
        ctk.CTkEntry(master=data_root, textvariable=self.email).grid(column=1, row=3, sticky=tk.W, padx=0, pady=15)

        # channel mask
        ctk.CTkLabel(master=data_root, text='Nuclear Mask', anchor='w').grid(column=0, row=4, sticky=tk.W, padx=15, pady=15)
        mask_combobox = ctk.CTkComboBox(master=data_root, values=["DAPI", "AF350", "Option3"], variable=self.mask_channel)
        mask_combobox['values'] = ["DAPI", "AF350", "Option3"]
        # prevent typing a value
        mask_combobox['state'] = 'readonly'
        mask_combobox.grid(column=1, row=4, sticky=tk.W, padx=0, pady=15)
        mask_combobox.get()

        #nuclei identification mode
        self.nuc_idnt_mode.set('unet')
        ctk.CTkLabel(master=data_root, text='Nuclei identification mode', anchor='w').grid(column=0, row=5, sticky=tk.W, padx=15, pady=15)
        modes_parent = ctk.CTkFrame(master=data_root, fg_color=None)
        modes_parent.grid(column=1, row=5, sticky=tk.W)
        modes_parent.columnconfigure(0, weight=1)
        modes_parent.columnconfigure(1, weight=1)

        # radio buttons
        r1 = ctk.CTkRadioButton(
            modes_parent,
            text='Machine learning',
            value='unet',
            variable=self.nuc_idnt_mode,
        )
        r1.grid(column=0, row=0, sticky=tk.W, padx=0, pady=15)
        r2 = ctk.CTkRadioButton(
            modes_parent,
            text='Threshold',
            value='thr',
            variable=self.nuc_idnt_mode
        )
        r2.grid(column=1, row=0, sticky=tk.W, padx=15, pady=15)

        # threshold slider
        self.thr.set(30)
        ctk.CTkLabel(master=data_root, text='Pixel Threshold', anchor='w').grid(column=0, row=6, sticky=tk.W, padx=15, pady=15)
        slider_parent = ctk.CTkFrame(master=data_root, fg_color=None)
        slider_parent.grid(column=1, row=6, sticky=tk.EW, padx=0, pady=15)
        slider = ctk.CTkSlider(
            slider_parent,
            from_=0,
            to=255,
            orient='horizontal',  # vertical
            width=250,
            command=self.slider_changed,
            variable=self.thr
        )

        slider.grid(column=0, row=0, sticky=tk.EW, padx=0, pady=15)
        # value label
        self.value_label = ctk.CTkLabel(slider_parent, text='{: .0f}'.format(self.thr.get()))
        self.value_label.grid(column=1, row=0, padx=15, pady=0, sticky=tk.E)


        #minimal nucleus area
        ctk.CTkLabel(master=data_root, text='Minimum nucleus area', anchor='w').grid(column=0, row=7, sticky=tk.W, padx=15, pady=15)
        spin_box = ctk.CTkEntry(master=data_root, width=50, textvariable=self.min_nuc_area)
        spin_box.grid(column=1, row=7, sticky=tk.W, padx=0, pady=15)

        #Separate touching cells

        separate_check = ctk.CTkCheckBox(master=data_root,
                                         text='Separate touching cells',
                                         variable=self.separating_cells,
                                         onvalue='true',
                                         offvalue='false')
        separate_check.grid(column=0, row=8, sticky=tk.W, padx=30, pady=15, columnspan=2)

        # Analyze perinuclear area
        perinuclear_check = ctk.CTkCheckBox(master=data_root,
                                         text='Analyze perinuclear area',
                                         variable=self.perinuclear_area,
                                         onvalue='true',
                                         offvalue='false')
        perinuclear_check.grid(column=0, row=9, sticky=tk.W, padx=30, pady=15, columnspan=2)

        self.analize_button = ctk.CTkButton(
            master=data_root,
            text="Analyze")
        self.analize_button.grid(column=1, row=9, sticky=tk.E, padx=15, pady=30)

        self.analize_button = ctk.CTkButton(
            master=data_root,
            text="Analyze")
        self.analize_button.grid(column=1, row=9, sticky=tk.E, padx=15, pady=30)

        #Analyze perinuclear area



    def slider_changed(self, event):
        self.value_label.configure(text='{: .0f}'.format(self.thr.get()))

    def input_button(self):
        self.input_folder.set(filedialog.askdirectory())

    def output_button(self):
        self.output_folder.set(filedialog.askdirectory())

# design customization
ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green

root = ctk.CTk()
root.title('MAL image analysis kit')
window_width = 350
window_height = 250

# get the screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# find the center point
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
root.iconbitmap('img/favicon.ico')

# First page: Analysis type
analysis_frame = ctk.CTkFrame(master=root)
analysis_page = AnalysisTypePage(analysis_frame)
analysis_frame.pack(expand=True)
analysis_page.continue_button.configure(command=show_data_page)

# Second page: Data collection
data_frame = ctk.CTkFrame(master=root)
a = analysis_page.selected_type.get()
data_page = DataCollectionPage(data_frame)
data_page.analize_button.configure(command=run_analysis)

root.mainloop()


