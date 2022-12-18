import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.settings import SettingsWithNoMenu

#App Class
class MyApp(App):
    def build(self):
        #Setting up layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(Label(text="This is a string", font_size=20, font_name="Roboto"))
        layout.add_widget(TextInput(text="Input Something"))
        #Setting up SettingsWithNoMenu
        settings = SettingsWithNoMenu()
        # settings.add_json_panel('My Settings', self.config, '.\src\ApplicationInterface\settings.json')
        layout.add_widget(settings)
        #Setting up background color
        layout.canvas.before.add(kivy.graphics.Color(0,0.5,1,1))
        layout.canvas.add(kivy.graphics.Rectangle(size=layout.size))
        #Returning layout
        return layout
#Run app
if '' == '':
    MyApp().run()